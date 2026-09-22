from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.labeling.labeler import export_ground_truth, load_dataset, progress, save_label

app = FastAPI(title="HDFC Bank News Labeling Tool")
TEMPLATE = Path(__file__).parent / "templates" / "labeler.html"


class LabelRequest(BaseModel):
    article_id: str
    manual_label: str | None = None
    review_notes: str = ""
    reviewed: bool = False
    confirm: bool = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


@app.get("/api/state")
def state() -> dict:
    data = load_dataset()
    records = data.to_dict(orient="records")
    first_unreviewed = next((index for index, row in enumerate(records) if str(row.get("reviewed", "")).lower() not in {"true", "1", "yes"}), 0)
    return {"articles": records, "progress": progress(data), "resume_index": first_unreviewed}


@app.post("/api/label")
def label(request: LabelRequest) -> dict:
    try:
        data = save_label(request.article_id, request.manual_label, request.review_notes, request.reviewed, request.confirm)
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"progress": progress(data), "articles": data.to_dict(orient="records")}


@app.post("/api/export")
def export() -> dict:
    _, report = export_ground_truth()
    return report


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("LABELER_PORT", "8002"))
    print("HDFC BANK NEWS LABELING TOOL")
    print("----------------------------")
    print("Dataset: data/processed/news/HDFCBANK_news_manual_validation.csv")
    print(f"Open: http://127.0.0.1:{port}")
    uvicorn.run("src.labeling.app:app", host="127.0.0.1", port=port, reload=False)
