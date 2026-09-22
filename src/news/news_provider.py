from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests

from src.config import ROOT_DIR, settings

logger = logging.getLogger(__name__)

NEWS_COLUMNS = [
    "article_id", "published_at", "date", "headline", "description", "content",
    "source_name", "author", "url", "symbol", "query", "provider", "fetched_at",
]
QUERIES = [
    '"HDFC Bank"', '"HDFC Bank Ltd"', "HDFCBANK", '"HDFC Bank NSE"',
    '"HDFC Bank stock"', '"HDFC Bank RBI"', '"HDFC Bank results"',
    '"HDFC Bank earnings"', '"HDFC Bank merger"', '"HDFC Bank management"',
    '"HDFC Bank loan growth"', '"HDFC Bank NIM"', '"HDFC Bank deposits"',
]


class NewsProvider(ABC):
    provider_name = "abstract"

    @abstractmethod
    def fetch_news(self, query: str, start_date: date, end_date: date) -> list[dict[str, Any]]:
        raise NotImplementedError

    def fetch_historical_news(self, queries: list[str], start_date: date, end_date: date) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for query in queries:
            records.extend(self.fetch_news(query, start_date, end_date))
        return records

    @abstractmethod
    def normalize_article(self, article: dict[str, Any], query: str) -> dict[str, Any]:
        raise NotImplementedError


class GoogleNewsRSSProvider(NewsProvider):
    provider_name = "Google News RSS"
    base_url = "https://news.google.com/rss/search"

    def __init__(self, symbol: str = "HDFCBANK", timeout: int = 30, retries: int = 3) -> None:
        self.symbol = symbol
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HDFC-Bank-News-Collector/1.0"})

    def fetch_news(self, query: str, start_date: date, end_date: date) -> list[dict[str, Any]]:
        url = f"{self.base_url}?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                records = []
                for item in root.findall("./channel/item"):
                    record = {element.tag.split("}")[-1]: (element.text or "").strip() for element in item}
                    published = self._parse_date(record.get("pubDate"))
                    if published is not None and start_date <= published.date() <= end_date:
                        records.append(record)
                return records
            except (requests.RequestException, ET.ParseError) as exc:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"{self.provider_name} failed for query {query!r}: {exc}") from exc
                time.sleep(2 ** attempt)
        return []

    @staticmethod
    def _parse_date(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    def normalize_article(self, article: dict[str, Any], query: str) -> dict[str, Any]:
        published = self._parse_date(article.get("pubDate"))
        published_at = published.astimezone(timezone.utc).isoformat() if published else None
        url = article.get("link") or None
        headline = re.sub(r"\s+", " ", (article.get("title") or "").strip())
        description = re.sub(r"\s+", " ", (article.get("description") or "").strip()) or None
        source = re.sub(r"\s+", " ", (article.get("source") or "").strip()) or None
        identity = url or f"{headline}|{published_at or ''}"
        article_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        fetched_at = datetime.now(timezone.utc).isoformat()
        return {
            "article_id": article_id,
            "published_at": published_at,
            "date": published.date().isoformat() if published else None,
            "headline": headline or None,
            "description": description,
            "content": None,
            "source_name": source,
            "author": None,
            "url": url,
            "symbol": self.symbol,
            "query": query,
            "provider": self.provider_name,
            "fetched_at": fetched_at,
        }


def normalize_articles(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, int]:
    frame = pd.DataFrame(records, columns=NEWS_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS), 0
    frame = frame[NEWS_COLUMNS].copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    frame["date"] = frame["published_at"].dt.date.astype("string")
    before = len(frame)
    frame = frame.drop_duplicates(subset=["article_id"], keep="first")
    frame = frame.drop_duplicates(subset=["url"], keep="first")
    frame = frame.drop_duplicates(subset=["headline", "date"], keep="first")
    frame = frame.sort_values("published_at", ascending=True, na_position="last").reset_index(drop=True)
    return frame, before - len(frame)


def fetch_hdfc_news(force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_dir = ROOT_DIR / "data" / "raw" / "news"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "HDFCBANK_news_raw.csv"
    metadata_path = raw_dir / "metadata.json"
    end_date = date.today()
    start_date = end_date - timedelta(days=5 * 365)
    provider = GoogleNewsRSSProvider(symbol=settings.ticker)
    existing = pd.DataFrame(columns=NEWS_COLUMNS)
    if raw_path.exists() and not force_refresh:
        existing = pd.read_csv(raw_path)
        logger.info("Existing news cache: %s rows", len(existing))
    fetch_start = start_date
    if not existing.empty and not force_refresh and "published_at" in existing.columns:
        cached_dates = pd.to_datetime(existing["published_at"], errors="coerce", utc=True).dropna()
        if not cached_dates.empty:
            fetch_start = max(start_date, (cached_dates.max().date() + timedelta(days=1)))
    queries = QUERIES
    notes = ["Google News RSS does not provide guaranteed historical pagination; actual returned coverage is reported."]
    try:
        fetched = []
        if fetch_start <= end_date:
            for query in queries:
                for article in provider.fetch_news(query, fetch_start, end_date):
                    fetched.append(provider.normalize_article(article, query))
        else:
            notes.append("Cache already reaches the current date; no newer RSS window was requested.")
        fresh, duplicates_removed = normalize_articles(fetched)
    except Exception as exc:
        metadata = {"company": "HDFC Bank Ltd", "symbol": settings.ticker, "data_type": "news", "providers": [provider.provider_name], "queries": queries, "requested_start_date": start_date.isoformat(), "requested_end_date": end_date.isoformat(), "actual_start_date": None, "actual_end_date": None, "article_count": 0, "duplicate_count_removed": 0, "downloaded_at": datetime.now(timezone.utc).isoformat(), "status": "failed", "notes": [str(exc)]}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    if not existing.empty:
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined, merged_duplicates = normalize_articles(combined.to_dict("records"))
        duplicates_removed += merged_duplicates
    else:
        combined = fresh
    if combined.empty:
        raise RuntimeError("No real HDFC Bank news articles were returned by the configured public provider.")
    combined.to_csv(raw_path, index=False)
    metadata = {
        "company": "HDFC Bank Ltd", "symbol": settings.ticker, "data_type": "news",
        "providers": [provider.provider_name], "queries": queries,
        "requested_start_date": start_date.isoformat(), "requested_end_date": end_date.isoformat(),
        "actual_start_date": str(combined["date"].min()), "actual_end_date": str(combined["date"].max()),
        "article_count": int(len(combined)), "duplicate_count_removed": int(duplicates_removed),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "success" if combined["date"].min() <= start_date.isoformat() else "partial",
        "notes": notes,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return combined, metadata
