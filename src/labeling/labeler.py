from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR

CLEAN_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_clean.csv"
BASELINE_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_baseline_sentiment.csv"
MANUAL_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_manual_validation.csv"
GROUND_TRUTH_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_ground_truth.csv"
VALIDATION_PATH = ROOT_DIR / "reports" / "validation" / "news_ground_truth_validation.json"
LABELS = {"POSITIVE", "NEUTRAL", "NEGATIVE"}
MANUAL_COLUMNS = [
    "article_id", "date", "headline", "clean_text", "source_name", "url",
    "suggested_label", "suggested_confidence", "manual_label", "review_notes",
    "reviewed", "reviewed_at",
]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(CLEAN_PATH).drop_duplicates("article_id").copy()
    baseline = pd.read_csv(BASELINE_PATH) if BASELINE_PATH.exists() else pd.DataFrame()
    if not baseline.empty:
        keep = [column for column in ["article_id", "sentiment", "confidence"] if column in baseline.columns]
        baseline = baseline[keep].drop_duplicates("article_id")
        baseline = baseline.rename(columns={"sentiment": "suggested_label", "confidence": "suggested_confidence"})
        clean = clean.merge(baseline, on="article_id", how="left")
    else:
        clean["suggested_label"] = ""
        clean["suggested_confidence"] = None
    for column in ["suggested_label", "suggested_confidence"]:
        if column not in clean.columns:
            clean[column] = "" if column == "suggested_label" else None
    return clean, baseline


def _sample_articles(clean: pd.DataFrame, target_size: int) -> pd.DataFrame:
    available = clean.drop_duplicates(subset=["article_id", "url", "headline"], keep="first").copy()
    if len(available) <= target_size:
        return available
    available["date_bucket"] = pd.qcut(available["date"].rank(method="first"), q=min(12, len(available)), labels=False, duplicates="drop")
    available["confidence_bucket"] = pd.qcut(available["suggested_confidence"].fillna(0.0).rank(method="first"), q=3, labels=False, duplicates="drop")
    groups = [group for _, group in available.groupby(["date_bucket", "source_name", "suggested_label", "confidence_bucket"], dropna=False)]
    selected: list[pd.DataFrame] = []
    for group in groups:
        selected.append(group.sample(n=1, random_state=42))
    result = pd.concat(selected, ignore_index=True).drop_duplicates("article_id")
    if len(result) < target_size:
        remaining = available.loc[~available["article_id"].isin(result["article_id"])]
        result = pd.concat([result, remaining.sample(n=min(target_size - len(result), len(remaining)), random_state=42)], ignore_index=True)
    return result.head(target_size)


def ensure_dataset(target_size: int = 300) -> pd.DataFrame:
    clean, _ = _load_inputs()
    sampled = _sample_articles(clean, target_size)
    existing = pd.read_csv(MANUAL_PATH) if MANUAL_PATH.exists() else pd.DataFrame()
    existing = existing.drop_duplicates("article_id", keep="last")
    if not existing.empty:
        existing_ids = set(existing["article_id"]) & set(clean["article_id"])
        missing_existing = clean[clean["article_id"].isin(existing_ids) & ~clean["article_id"].isin(sampled["article_id"])]
        sampled = pd.concat([sampled, missing_existing], ignore_index=True).drop_duplicates("article_id", keep="first")
        if len(sampled) > target_size:
            keep_ids = set(existing_ids)
            preserved_rows = sampled[sampled["article_id"].isin(keep_ids)]
            additional = sampled[~sampled["article_id"].isin(keep_ids)].head(max(0, target_size - len(preserved_rows)))
            sampled = pd.concat([preserved_rows, additional], ignore_index=True)
    if not existing.empty:
        preserved = existing.set_index("article_id")
    else:
        preserved = pd.DataFrame().set_index(pd.Index([], name="article_id"))
    rows = []
    for _, article in sampled.iterrows():
        old = preserved.loc[article["article_id"]].to_dict() if article["article_id"] in preserved.index else {}
        rows.append({
            "article_id": article["article_id"], "date": article.get("date"), "headline": article.get("headline", ""),
            "clean_text": article.get("clean_text", ""), "source_name": article.get("source_name", ""), "url": article.get("url", ""),
            "suggested_label": article.get("suggested_label", "") if pd.notna(article.get("suggested_label", "")) else "",
            "suggested_confidence": article.get("suggested_confidence") if pd.notna(article.get("suggested_confidence")) else None,
            "manual_label": old.get("manual_label", "") or "", "review_notes": old.get("review_notes", "") or "",
            "reviewed": bool(_truthy(old.get("reviewed", False))), "reviewed_at": old.get("reviewed_at", "") or "",
        })
    result = pd.DataFrame(rows, columns=MANUAL_COLUMNS).sort_values("date").reset_index(drop=True)
    MANUAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(MANUAL_PATH, index=False)
    return result


def load_dataset() -> pd.DataFrame:
    data = ensure_dataset()
    for column in ["manual_label", "review_notes", "reviewed_at"]:
        data[column] = data[column].fillna("").astype(object)
    data["reviewed"] = data["reviewed"].map(_truthy)
    return data


def save_label(article_id: str, manual_label: str | None, review_notes: str = "", reviewed: bool = False, confirm: bool = False) -> pd.DataFrame:
    if manual_label and manual_label not in LABELS:
        raise ValueError(f"manual_label must be one of {sorted(LABELS)}")
    data = load_dataset()
    matches = data.index[data["article_id"].eq(article_id)].tolist()
    if not matches:
        raise KeyError(f"Unknown article_id: {article_id}")
    index = matches[0]
    old_label = str(data.at[index, "manual_label"] or "").strip()
    if old_label and old_label != (manual_label or "") and not confirm:
        raise PermissionError("Changing an existing label requires confirmation.")
    data.at[index, "manual_label"] = manual_label or ""
    data.at[index, "review_notes"] = review_notes
    data.at[index, "reviewed"] = bool(reviewed and manual_label)
    data.at[index, "reviewed_at"] = datetime.now(timezone.utc).isoformat() if data.at[index, "reviewed"] else ""
    data.to_csv(MANUAL_PATH, index=False)
    return data


def progress(data: pd.DataFrame) -> dict:
    reviewed = data["reviewed"].map(_truthy) & data["manual_label"].isin(LABELS)
    counts = data.loc[reviewed, "manual_label"].value_counts().to_dict()
    return {"total": len(data), "reviewed": int(reviewed.sum()), "remaining": int(len(data) - reviewed.sum()), "positive": int(counts.get("POSITIVE", 0)), "neutral": int(counts.get("NEUTRAL", 0)), "negative": int(counts.get("NEGATIVE", 0)), "skipped": int((~reviewed).sum())}


def export_ground_truth() -> tuple[pd.DataFrame, dict]:
    data = load_dataset()
    reviewed = data[data["reviewed"].map(_truthy) & data["manual_label"].isin(LABELS)].copy()
    errors = []
    errors.extend(["duplicate article_id"] if reviewed["article_id"].duplicated().any() else [])
    errors.extend(["empty text"] if reviewed["clean_text"].fillna("").str.strip().eq("").any() else [])
    errors.extend(["invalid labels"] if (~reviewed["manual_label"].isin(LABELS)).any() else [])
    output = reviewed[["article_id", "date", "headline", "clean_text", "manual_label"]].copy()
    if not errors:
        output.to_csv(GROUND_TRUTH_PATH, index=False)
    counts = output["manual_label"].value_counts().to_dict()
    report = {"total_source_articles": len(data), "sampled_articles": len(data), "reviewed_articles": len(output), "skipped_articles": len(data) - len(output), "positive_count": int(counts.get("POSITIVE", 0)), "neutral_count": int(counts.get("NEUTRAL", 0)), "negative_count": int(counts.get("NEGATIVE", 0)), "class_percentages": {label: round(counts.get(label, 0) / len(output) * 100, 2) if len(output) else 0.0 for label in sorted(LABELS)}, "duplicate_count": int(output["article_id"].duplicated().sum()), "missing_text_count": int(output["clean_text"].fillna("").str.strip().eq("").sum()), "invalid_label_count": int((~output["manual_label"].isin(LABELS)).sum()), "validation_status": "PASS" if not errors else "FAIL", "errors": errors, "output": str(GROUND_TRUTH_PATH)}
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output, report
