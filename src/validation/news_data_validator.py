from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "article_id", "published_at", "date", "headline", "description", "content",
    "source_name", "author", "url", "symbol", "query", "provider", "fetched_at",
]


def validate_news_data(data: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return {"status": "FAIL", "errors": errors, "warnings": warnings, "article_count": 0}
    if data.empty:
        errors.append("Dataset is empty.")
    parsed_dates = pd.to_datetime(data["published_at"], errors="coerce", utc=True)
    if parsed_dates.isna().any():
        errors.append(f"Unparseable published_at values: {int(parsed_dates.isna().sum())}")
    if not data["published_at"].is_monotonic_increasing:
        warnings.append("Rows are not sorted by published_at.")
    if data["article_id"].dropna().duplicated().any():
        errors.append("Duplicate article_id values found.")
    nonempty_urls = data["url"].dropna().astype(str)
    invalid_urls = [url for url in nonempty_urls if urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc]
    if invalid_urls:
        errors.append(f"Invalid URLs: {len(invalid_urls)}")
    if data["url"].dropna().duplicated().any():
        warnings.append("Duplicate URLs found.")
    if data["headline"].fillna("").astype(str).str.strip().eq("").any():
        errors.append("Empty headlines found.")
    exact_duplicates = int(data.duplicated().sum())
    if exact_duplicates:
        errors.append(f"Exact duplicate records found: {exact_duplicates}")
    search_text = (data["headline"].fillna("") + " " + data["description"].fillna("")).str.lower()
    relevant = search_text.str.contains(r"hdfc\s*bank|hdfcbank", regex=True, na=False)
    if (~relevant).any():
        query_qualified = data["query"].fillna("").astype(str).str.contains("HDFC Bank|HDFCBANK", case=False, regex=True)
        unresolved = (~relevant) & (~query_qualified)
        if unresolved.any():
            errors.append(f"Articles not clearly related to HDFC Bank: {int(unresolved.sum())}")
        else:
            warnings.append(f"{int((~relevant).sum())} articles are query-qualified but do not repeat HDFC Bank in the headline/description.")
    if data["symbol"].dropna().astype(str).ne("HDFCBANK").any():
        errors.append("Unexpected symbol value found.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "article_count": int(len(data)),
        "first_article": parsed_dates.min().isoformat() if not parsed_dates.dropna().empty else None,
        "last_article": parsed_dates.max().isoformat() if not parsed_dates.dropna().empty else None,
        "duplicate_article_ids": int(data["article_id"].duplicated().sum()),
        "duplicate_urls": int(data["url"].dropna().duplicated().sum()),
        "exact_duplicate_records": exact_duplicates,
    }
