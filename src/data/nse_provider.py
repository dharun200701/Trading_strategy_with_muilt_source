from __future__ import annotations

import json
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from urllib.error import HTTPError
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from nsepython import get_bhavcopy, nsefetch

from src.config import ROOT_DIR, settings

logger = logging.getLogger(__name__)


class NSEProvider:
    raw_filename = "HDFCBANK_NSE_raw.csv"

    def __init__(self, symbol: str = "HDFCBANK", force_refresh: bool = False) -> None:
        self.symbol = symbol
        self.force_refresh = force_refresh
        self.raw_dir = ROOT_DIR / "data" / "raw" / "market"
        self.processed_dir = ROOT_DIR / "data" / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def raw_path(self) -> Path:
        return self.raw_dir / self.raw_filename

    def _inspect_quote_response(self) -> None:
        url = (
            "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?"
            f"functionName=getSymbolData&marketType=N&series=EQ&symbol={self.symbol}"
        )
        payload = nsefetch(url)
        records = payload.get("equityResponse", []) if isinstance(payload, dict) else []
        logger.info(
            "NSE quote diagnostic: response_type=%s top_level_keys=%s records=%s first_record_keys=%s",
            type(payload).__name__,
            list(payload.keys()) if isinstance(payload, dict) else [],
            len(records) if isinstance(records, list) else 0,
            list(records[0].keys()) if isinstance(records, list) and records else [],
        )

    @staticmethod
    def _date_ranges(start: date, end: date, chunk_days: int = 180) -> list[tuple[date, date]]:
        ranges: list[tuple[date, date]] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
            ranges.append((cursor, chunk_end))
            cursor = chunk_end + timedelta(days=1)
        return ranges

    def _fetch_range_diagnostic(self, start: date, end: date) -> pd.DataFrame:
        url = (
            "https://www.nseindia.com/api/historical/securityArchives?"
            f"from={start.strftime('%d-%m-%Y')}&to={end.strftime('%d-%m-%Y')}"
            f"&symbol={self.symbol}&dataType=priceVolumeDeliverable&series=EQ"
        )
        payload = nsefetch(url)
        records = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(records, list) and records:
            logger.info(
                "NSE historical diagnostic: range=%s..%s records=%s first_record_keys=%s",
                start, end, len(records), list(records[0].keys()),
            )
            return pd.DataFrame.from_records(records)
        logger.info(
            "NSE historical diagnostic: range=%s..%s response_type=%s top_level_keys=%s records=0",
            start, end, type(payload).__name__, list(payload.keys()) if isinstance(payload, dict) else [],
        )
        return pd.DataFrame()

    def _fetch_archive_day(self, day: date, retries: int = 2) -> tuple[date, pd.DataFrame | None, str | None]:
        for attempt in range(retries + 1):
            try:
                daily = get_bhavcopy(day.strftime("%d-%m-%Y"))
                if not isinstance(daily, pd.DataFrame) or daily.empty:
                    return day, None, "empty response"
                daily = daily.copy()
                daily.columns = [str(column).strip() for column in daily.columns]
                required_source = {"SYMBOL", "SERIES"}
                if not required_source.issubset(daily.columns):
                    return day, None, f"missing source fields: {sorted(required_source - set(daily.columns))}"
                daily = daily[
                    (daily["SYMBOL"].astype(str).str.upper() == self.symbol)
                    & (daily["SERIES"].astype(str).str.strip() == "EQ")
                ]
                return day, daily if not daily.empty else None, None
            except UnicodeDecodeError:
                # A small number of older NSE archive files contain a non-UTF8 byte.
                # Keep the same NSE archive URL and use a compatible CSV decoding only
                # for that response; nsepython remains the primary fetch mechanism.
                url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{day.strftime('%d%m%Y')}.csv"
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    if response.content[:2] == b"PK":
                        daily = pd.read_excel(io.BytesIO(response.content), engine="openpyxl")
                    else:
                        daily = pd.read_csv(io.BytesIO(response.content), encoding="latin1")
                    daily.columns = [str(column).strip() for column in daily.columns]
                    daily = daily[
                        (daily["SYMBOL"].astype(str).str.upper() == self.symbol)
                        & (daily["SERIES"].astype(str).str.strip() == "EQ")
                    ]
                    return day, daily if not daily.empty else None, None
                except Exception as exc:
                    if attempt == retries:
                        return day, None, f"non-UTF8 NSE archive fallback failed: {exc}"
            except HTTPError as exc:
                if exc.code == 404:
                    return day, None, "no NSE archive published"
                if attempt == retries:
                    return day, None, f"HTTP {exc.code}: {exc.reason}"
            except Exception as exc:
                if attempt == retries:
                    return day, None, str(exc)
            time.sleep(0.5 * (attempt + 1))
        return day, None, "request failed after retries"

    def _fetch_archive_chunk(self, start: date, end: date, diagnostic: bool = False) -> pd.DataFrame:
        trading_days = [
            start + timedelta(days=offset)
            for offset in range((end - start).days + 1)
            if (start + timedelta(days=offset)).weekday() < 5
        ]

        rows: list[pd.DataFrame] = []
        unavailable_days: list[str] = []
        failed_days: list[str] = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self._fetch_archive_day, day) for day in trading_days]
            for future in as_completed(futures):
                day, daily, error = future.result()
                if daily is not None:
                    rows.append(daily)
                elif error == "no NSE archive published":
                    unavailable_days.append(str(day))
                elif error:
                    failed_days.append(f"{day}: {error}")

        if diagnostic:
            sample = rows[0].iloc[0].to_dict() if rows else None
            logger.info(
                "NSE archive diagnostic: status=received response_type=DataFrame records=%s sample=%s fields=%s",
                sum(len(row) for row in rows), sample, list(rows[0].columns) if rows else [],
            )
        if failed_days:
            preview = "; ".join(failed_days[:3])
            suffix = f"; ... {len(failed_days) - 3} more" if len(failed_days) > 3 else ""
            logger.warning("NSE archive requests failed for %s..%s: %s%s", start, end, preview, suffix)
            raise RuntimeError(f"NSE archive requests failed for {start}..{end}: {preview}{suffix}")
        if unavailable_days:
            logger.info("NSE archive had no published file for %s non-trading dates in %s..%s.", len(unavailable_days), start, end)
        if not rows:
            raise RuntimeError(f"NSE returned no HDFCBANK archive records for {start}..{end}.")
        return pd.concat(rows, ignore_index=True)

    def _download_historical_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        logger.info("Fetching NSE historical data for %s from %s to %s", self.symbol, start_date, end_date)
        combined: list[pd.DataFrame] = []
        ranges = self._date_ranges(start_date, end_date)
        for index, (range_start, range_end) in enumerate(ranges, start=1):
            logger.info("Fetching missing NSE chunk %s/%s: %s..%s", index, len(ranges), range_start, range_end)
            range_data = self._fetch_archive_chunk(range_start, range_end, diagnostic=index == 1)
            combined.append(range_data)
        if not combined:
            raise RuntimeError("NSE returned no historical HDFCBANK data.")
        return pd.concat(combined, ignore_index=True)

    def _normalize_and_validate(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data.columns = [str(column).strip() for column in data.columns]
        data = data.rename(columns={
            "SYMBOL": "symbol", "SERIES": "series", "DATE1": "date",
            "OPEN_PRICE": "open", "HIGH_PRICE": "high", "LOW_PRICE": "low",
            "CLOSE_PRICE": "close", "PREV_CLOSE": "previous_close",
            "TTL_TRD_QNTY": "volume", "TURNOVER_LACS": "turnover_lacs",
            "LAST_PRICE": "last_price", "AVG_PRICE": "average_price",
            "DELIV_QTY": "delivery_quantity", "DELIV_PER": "delivery_percentage",
        })
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise RuntimeError(f"NSE historical response is missing required fields: {missing}")
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        for column in ["open", "high", "low", "close", "previous_close", "volume", "turnover_lacs"]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        before = len(data)
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        data = data[
            (data["open"] > 0) & (data["high"] > 0) & (data["low"] > 0)
            & (data["close"] > 0) & (data["volume"] >= 0) & (data["high"] >= data["low"])
            & (data["open"] >= data["low"]) & (data["open"] <= data["high"])
            & (data["close"] >= data["low"]) & (data["close"] <= data["high"])
        ]
        data = data.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        removed = before - len(data)
        if removed:
            logger.warning("Removed %s invalid or duplicate NSE rows during validation.", removed)
        if data.empty:
            raise RuntimeError("NSE historical data contained no valid HDFCBANK rows after validation.")
        return data

    def download_historical_data(self) -> pd.DataFrame:
        metadata_path = self.raw_dir / "metadata.json"
        end_date = date.today()
        requested_start = end_date - timedelta(days=5 * 365)
        existing_rows = 0
        new_rows = 0
        chunks_requested = 0
        chunks_successful = 0
        chunks_failed = 0

        existing = pd.DataFrame()
        if self.raw_path.exists():
            existing = self._normalize_and_validate(pd.read_csv(self.raw_path))
            existing_rows = len(existing)
            logger.info("Existing rows: %s", existing_rows)
            logger.info("Existing coverage: %s to %s", existing["date"].min().date(), existing["date"].max().date())

        missing_ranges: list[tuple[date, date]] = []
        if self.force_refresh or existing.empty:
            missing_ranges = [(requested_start, end_date)]
        else:
            existing_start = existing["date"].min().date()
            existing_end = existing["date"].max().date()
            if existing_start > requested_start:
                missing_ranges.append((requested_start, existing_start - timedelta(days=1)))
            if existing_end < end_date:
                missing_ranges.append((existing_end + timedelta(days=1), end_date))

        if not missing_ranges:
            logger.info("Cached NSE data already covers the requested five-year range.")
            return existing

        logger.info("Missing coverage: %s to %s", missing_ranges[0][0], missing_ranges[-1][1])
        fetched: list[pd.DataFrame] = []
        try:
            for missing_start, missing_end in missing_ranges:
                ranges = self._date_ranges(missing_start, missing_end)
                chunks_requested += len(ranges)
                for range_start, range_end in ranges:
                    chunk = self._fetch_archive_chunk(range_start, range_end, diagnostic=not fetched)
                    fetched.append(chunk)
                    chunks_successful += 1
            new_rows = sum(len(chunk) for chunk in fetched)
            normalized_fetched = [self._normalize_and_validate(chunk) for chunk in fetched]
            merged = self._normalize_and_validate(pd.concat([existing, *normalized_fetched], ignore_index=True))
        except Exception as exc:
            chunks_failed += 1
            logger.error("NSE incremental download failed: %s", exc)
            raise RuntimeError(f"Unable to extend HDFCBANK NSE data; existing CSV preserved. {exc}") from exc

        temporary_path = self.raw_dir / "HDFCBANK_NSE_raw.tmp.csv"
        merged.to_csv(temporary_path, index=False)
        temporary_path.replace(self.raw_path)
        metadata = {
            "source": "NSE India", "provider": "nsepython", "symbol": self.symbol, "series": "EQ",
            "frequency": "daily", "requested_start_date": requested_start.isoformat(),
            "requested_end_date": end_date.isoformat(), "actual_first_date": merged["date"].min().strftime("%Y-%m-%d"),
            "actual_last_date": merged["date"].max().strftime("%Y-%m-%d"), "existing_rows": existing_rows,
            "newly_downloaded_rows": new_rows, "final_row_count": len(merged),
            "download_timestamp_utc": datetime.utcnow().isoformat(), "validation_status": "PASSED",
            "chunks_requested": chunks_requested, "chunks_successful": chunks_successful,
            "chunks_failed": chunks_failed,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Final dataset: rows=%s first=%s last=%s duplicate_dates=%s validation=PASSED", len(merged), merged["date"].min().date(), merged["date"].max().date(), merged["date"].duplicated().sum())
        return merged


def fetch_hdfc_market_data(force_refresh: bool = False) -> pd.DataFrame:
    return NSEProvider(symbol=settings.ticker, force_refresh=force_refresh).download_historical_data()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    data = fetch_hdfc_market_data(force_refresh=True)
    logger.info("Downloaded %s rows of NSE HDFC Bank data.", len(data))
