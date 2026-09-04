"""Vercel-compatible API for the Myntra Wishlist Maturation Layer MVP."""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "products.json"
DB_PATH = Path("/tmp/analytics.db") if os.getenv("VERCEL") else BASE_DIR / "analytics.db"

app = FastAPI(title="Myntra Wishlist Verdict MVP")


class Event(BaseModel):
    product_id: str
    event_name: str
    metadata: dict[str, Any] = {}


def products() -> list[dict[str, Any]]:
    items = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for item in items:
        if str(item["image"]).endswith("emerald_kurta.png"):
            item["image"] = "/api/images/emerald-kurta"
    return items


def product(product_id: str) -> dict[str, Any]:
    return next((item for item in products() if item["id"] == product_id), None) or _missing(product_id)


def _missing(product_id: str) -> None:
    raise HTTPException(status_code=404, detail=f"Unknown product: {product_id}")


def diagnose(item: dict[str, Any]) -> tuple[str, str]:
    signal = item["signals"]
    scores = {
        "Fit doubt": signal.get("size_chart_seconds", 0) * 1.4 + signal.get("image_zooms", 0) * 8,
        "Review gap": signal.get("review_scroll_percent", 0) * 1.2 + signal.get("pdp_revisits", 0) * 7,
        "Comparison paralysis": signal.get("similar_products_viewed", 0) * 14 + signal.get("pdp_revisits", 0) * 5,
        "Price doubt": (16 if item["price"] >= 3000 else 8 if item["price"] >= 2000 else 0) + signal.get("similar_products_viewed", 0) * 5,
    }
    hesitation = max(scores, key=scores.get)
    reasons = {
        "Fit doubt": "Size-chart time and image zooming suggest uncertainty about fit.",
        "Review gap": "Deep review reading and repeat visits suggest a need for stronger buyer validation.",
        "Comparison paralysis": "Multiple similar products and repeat visits signal difficulty choosing.",
        "Price doubt": "The shopper is still comparing value before committing.",
    }
    return hesitation, reasons[hesitation]


def confidence(item: dict[str, Any]) -> int:
    score = 58 + len(item.get("reviews", [])) * 4 + len(item.get("similar_buyer_notes", [])) * 5
    score += min(item.get("wishlist_age_hours", 0), 72) * 0.15
    notes = " ".join(item.get("return_notes", [])).lower()
    reviews = " ".join(item.get("reviews", [])).lower()
    if "high returns" in notes or "higher" in notes:
        score -= 8
    if "low return" in notes or "low size return" in notes:
        score += 6
    if "true to size" in reviews:
        score += 4
    return int(max(40, min(92, round(score))))


def case_file(item: dict[str, Any]) -> dict[str, Any]:
    hesitation, explanation = diagnose(item)
    score = confidence(item)
    label = "Strong buy confidence" if score >= 78 else "Buy with caution" if score >= 62 else "Wait / compare"
    action = "Add to cart or buy now with confidence." if score >= 78 else "Check the highlighted watch-outs before buying."
    return {"hesitation": hesitation, "explanation": explanation, "score": score, "label": label,
            "recommended_action": action, "evidence_strength": len(item["reviews"]) + len(item["return_notes"]) + len(item["similar_buyer_notes"]),
            "watch_outs": item["return_notes"], "positive_signals": item["similar_buyer_notes"]}


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts TEXT, product_id TEXT, event_name TEXT, metadata TEXT)")
    return conn


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    return FileResponse(BASE_DIR / "public" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/products")
def list_products() -> list[dict[str, Any]]:
    return products()


@app.get("/api/products/{product_id}")
def get_product(product_id: str) -> dict[str, Any]:
    item = product(product_id)
    return {"product": item, "case_file": case_file(item)}


@app.get("/api/compare/{product_id}")
def compare(product_id: str) -> dict[str, Any]:
    primary = product(product_id)
    target_id = primary.get("comparison_target_id")
    candidate = product(target_id) if target_id else next((x for x in products() if x["id"] != product_id and x["category"] == primary["category"]), None)
    if not candidate:
        raise HTTPException(status_code=404, detail="No comparison candidate available")
    primary_score, candidate_score = confidence(primary), confidence(candidate)
    winner, loser = (primary, candidate) if primary_score >= candidate_score else (candidate, primary)
    return {"winner": winner, "loser": loser, "winner_score": max(primary_score, candidate_score), "loser_score": min(primary_score, candidate_score),
            "reason": "stronger review, return, and similar-buyer evidence"}


@app.post("/api/events")
def record_event(event: Event) -> dict[str, str]:
    product(event.product_id)
    conn = db()
    conn.execute("INSERT INTO events (ts, product_id, event_name, metadata) VALUES (?, ?, ?, ?)", (datetime.utcnow().isoformat(), event.product_id, event.event_name, json.dumps(event.metadata)))
    conn.commit()
    conn.close()
    return {"status": "recorded"}


@app.get("/api/analytics")
def analytics() -> dict[str, int]:
    conn = db()
    rows = conn.execute("SELECT event_name, COUNT(*) FROM events GROUP BY event_name").fetchall()
    conn.close()
    counts = dict(rows)
    return {"verdict_views": counts.get("verdict_viewed", 0), "add_to_cart": counts.get("add_to_cart", 0), "buy_now": counts.get("buy_now", 0), "comparisons": counts.get("comparison_opened", 0)}


@app.get("/api/images/emerald-kurta")
def emerald_kurta() -> FileResponse:
    return FileResponse(BASE_DIR / "data" / "emerald_kurta.png")
