import json
import os
import sqlite3
import base64
import mimetypes
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from groq import Groq
except Exception:
    Groq = None

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "products.json"
DB_PATH = BASE_DIR / "analytics.db"

VIEW_WISHLIST = "Wishlist"
VIEW_PRODUCT = "Product Page"
VIEW_VERDICT = "Verdict Card"
VIEW_ANALYTICS = "Analytics"
VIEW_COMPARISON = "Comparison Analysis"
FEATURED_PRODUCT_ID = "MYN-002"

load_dotenv()

st.set_page_config(
    page_title="Myntra Wishlist Verdict MVP",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_products() -> List[Dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def image_src(value: str) -> str:
    if value.startswith("http://") or value.startswith("https://") or value.startswith("data:"):
        return value
    path = Path(value)
    if not path.exists():
        return value
    mime_type, _ = mimetypes.guess_type(str(path))
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:{0};base64,{1}".format(mime_type or "image/png", encoded)


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            product_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            metadata TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_event(product_id: str, event_name: str, metadata: Optional[Dict] = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO events (ts, product_id, event_name, metadata) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), product_id, event_name, json.dumps(metadata or {})),
    )
    conn.commit()
    conn.close()


def get_events() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM events ORDER BY id DESC", conn)
    conn.close()
    return df


def diagnose_hesitation(signals: Dict, product: Optional[Dict] = None) -> Tuple[str, str]:
    size = signals.get("size_chart_seconds", 0)
    review = signals.get("review_scroll_percent", 0)
    zooms = signals.get("image_zooms", 0)
    similar = signals.get("similar_products_viewed", 0)
    revisits = signals.get("pdp_revisits", 0)
    price = int(product.get("price", 0)) if product else 0
    premium_signal = 16 if price >= 3000 else 8 if price >= 2000 else 0

    scores = {
        "Fit Doubt": size * 1.4 + zooms * 8,
        "Review Gap": review * 1.2 + revisits * 7,
        "Fabric / Quality Doubt": zooms * 13 + review * 0.6,
        "Comparison Paralysis": similar * 14 + revisits * 5,
        "Styling Doubt": similar * 6 + zooms * 5,
        "Price Doubt": premium_signal + similar * 5 + revisits * 9 + max(0, 55 - review) * 0.5,
    }
    top = max(scores, key=scores.get)

    explanations = {
        "Fit Doubt": "The shopper spent unusually long on size information and zoomed into product imagery, which points to fit uncertainty.",
        "Review Gap": "The shopper scrolled deeply through reviews and returned to the PDP, which suggests they need stronger buyer validation.",
        "Fabric / Quality Doubt": "Repeated zooming and review consumption suggest uncertainty about whether the product will match the catalog promise.",
        "Comparison Paralysis": "The shopper viewed many similar products and revisited this item, which signals difficulty choosing among alternatives.",
        "Styling Doubt": "The shopper explored similar products and images, which suggests uncertainty about how to use or style the item.",
        "Price Doubt": "The shopper kept comparing and revisiting the item without building enough confidence, which suggests uncertainty about whether the price feels worth it.",
    }
    return top, explanations[top]


def evidence_strength(product: Dict) -> int:
    signals = product["signals"]
    strength = (
        len(product.get("reviews", [])) * 8
        + len(product.get("return_notes", [])) * 8
        + len(product.get("similar_buyer_notes", [])) * 10
        + min(product.get("wishlist_age_hours", 0), 72) * 0.6
        + signals.get("review_scroll_percent", 0) * 0.2
    )
    return int(round(strength))


def maturity_status(product: Dict) -> str:
    return "Verdict Ready" if evidence_strength(product) >= 110 else "Building Verdict"


def confidence_score(product: Dict, hesitation: str) -> int:
    base = 58
    review_count = len(product.get("reviews", []))
    buyer_notes = len(product.get("similar_buyer_notes", []))
    age = min(product.get("wishlist_age_hours", 0), 72)

    score = base + review_count * 4 + buyer_notes * 5 + age * 0.15

    notes = " ".join(product.get("return_notes", [])).lower()
    reviews_text = " ".join(product.get("reviews", [])).lower()
    if "high returns" in notes or "higher" in notes:
        score -= 8
    if "low return" in notes or "low size return" in notes:
        score += 6
    if "repeat-purchase" in notes or "repeat purchase" in notes:
        score += 5
    if "true to size" in reviews_text or "size is mostly true" in reviews_text:
        score += 4
    if "looks premium for the price" in reviews_text or "good value" in reviews_text:
        score += 3
    if hesitation == "Comparison Paralysis":
        score -= 10
    if hesitation == "Fit Doubt" and "fit" in notes and "high" in notes:
        score -= 7

    return int(max(40, min(92, round(score))))


def verdict_label(score: int) -> str:
    if score >= 78:
        return "Strong Buy Confidence"
    if score >= 62:
        return "Buy with Caution"
    return "Wait / Compare"


def fallback_case_file(product: Dict, hesitation: str, score: int) -> Dict:
    reviews = product.get("reviews", [])
    return_notes = product.get("return_notes", [])
    buyer_notes = product.get("similar_buyer_notes", [])
    similar = product.get("similar_saved_items", [])

    return {
        "fit_summary": _sentence_for_fit(product, hesitation, return_notes, buyer_notes),
        "positive_signals": _positive_signals(product, score),
        "review_summary": "Buyers generally say: " + " ".join(reviews[:2]),
        "styling_use_case": _sentence_for_styling(product),
        "comparison_note": "Compared with {0}, this item is strongest when the shopper wants {1} with lower decision effort.".format(
            ", ".join(similar[:3]),
            product["category"].lower(),
        ),
        "watch_outs": _watch_outs(return_notes, reviews),
        "recommended_action": _recommended_action(score),
    }


def _sentence_for_fit(product: Dict, hesitation: str, return_notes: List[str], buyer_notes: List[str]) -> str:
    notes = " ".join(return_notes + buyer_notes).lower()
    if "size up" in notes or "high returns" in notes or "fit mismatch" in notes:
        return "Fit confidence is moderate. Similar buyers showed some fit mismatch or size exchange behavior, so size selection should be double-checked."
    if "low return" in notes or "true to size" in " ".join(product.get("reviews", [])).lower():
        return "Fit confidence is relatively strong. Buyer evidence suggests the item is true to size for most shoppers."
    return "Fit confidence is usable but not perfect. The verdict should call out buyer fit notes before pushing the purchase."


def _sentence_for_styling(product: Dict) -> str:
    category = product.get("category", "")
    if category in ["Dresses", "Ethnic Wear"]:
        return "Best positioned for occasion-led use such as brunch, festive plans, dinner, or semi-formal events depending on the wardrobe need."
    if category == "Footwear":
        return "Best positioned as a versatile casual item that can pair with jeans, dresses, and everyday outfits."
    if category == "Western Wear":
        return "Best positioned for office, smart casual, and layered looks where the shopper needs extra confidence to justify the spend."
    return "Best positioned for practical repeat use where the shopper wants low-risk utility rather than heavy styling support."


def _positive_signals(product: Dict, score: int) -> str:
    notes = " ".join(product.get("return_notes", [])).lower()
    reviews_text = " ".join(product.get("reviews", [])).lower()
    signals: List[str] = []

    if "low return" in notes or "low size return" in notes:
        signals.append("low observed return friction")
    if "true to size" in reviews_text or "size is mostly true" in reviews_text:
        signals.append("buyers report reliable size confidence")
    if "repeat-purchase" in notes or "repeat purchase" in notes:
        signals.append("repeat-purchase behavior signals trust")
    if "looks premium for the price" in reviews_text or "good value" in reviews_text:
        signals.append("value perception is stronger than average")
    if "high keep rate" in notes or "kept this item often" in notes:
        signals.append("keep-rate signals are healthy")

    if signals and score >= 62:
        return "Positive signals: " + ", ".join(signals[:3]) + "."
    if score >= 78:
        return "Positive signals: buyer evidence is strong enough to support a confident recommendation."
    return "Positive signals are still forming, so the MVP should lean on caution rather than a hard recommendation."


def _watch_outs(return_notes: List[str], reviews: List[str]) -> str:
    joined = " ".join(return_notes + reviews)
    watch_phrases = []
    for marker in ["tight", "thin", "heavy", "creases", "oversized", "price", "wide feet", "fit mismatch"]:
        if marker.lower() in joined.lower():
            watch_phrases.append(marker)
    if watch_phrases:
        return "Watch-outs: " + ", ".join(sorted(set(watch_phrases))) + "."
    return "No major watch-outs found in the available evidence."


def _recommended_action(score: int) -> str:
    if score >= 78:
        return "Add to cart or buy now with confidence."
    if score >= 62:
        return "Buy only after checking the highlighted fit and quality watch-outs."
    return "Wait or compare against similar saved products before adding to bag."


def generate_llm_case_file(product: Dict, hesitation: str, score: int) -> Optional[Dict]:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key or Groq is None:
        return None

    try:
        client = Groq(api_key=api_key)
        prompt = f"""
You are powering a Myntra wishlist MVP.
Create a concise JSON case file for a saved product.

Product:
{json.dumps(product, indent=2)}

Diagnosed hesitation: {hesitation}
Confidence score: {score}

Return only valid JSON with these keys:
fit_summary, positive_signals, review_summary, styling_use_case, comparison_note, watch_outs, recommended_action.
Keep each value under 35 words.
Do not invent discounts. Do not use urgency tricks.
If the product is recommendation-worthy, include concrete positive buyer signals.
"""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return strict JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`").replace("json", "", 1).strip()
        return json.loads(content)
    except Exception as exc:
        st.session_state["last_llm_error"] = str(exc)
        return None


def get_case_file(product: Dict) -> Tuple[str, str, int, str, Dict]:
    hesitation, explanation = diagnose_hesitation(product["signals"], product)
    score = confidence_score(product, hesitation)
    if maturity_status(product) != "Verdict Ready":
        score = min(score, 61)
    label = verdict_label(score)
    case = generate_llm_case_file(product, hesitation, score) or fallback_case_file(product, hesitation, score)
    return hesitation, explanation, score, label, case


def pricing_snapshot(product: Dict) -> Tuple[int, int, int]:
    discount_map = {
        "MYN-001": 46,
        "MYN-002": 38,
        "MYN-003": 42,
        "MYN-004": 51,
        "MYN-005": 33,
        "MYN-006": 40,
    }
    price = int(product["price"])
    discount = discount_map.get(product["id"], 35)
    mrp = int(round(price / (1 - (discount / 100))))
    return price, mrp, discount


def short_title(text: str, limit: int = 38) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def html_bullets(items: List[str]) -> str:
    if not items:
        return "<li>No evidence available yet.</li>"
    return "".join("<li>{0}</li>".format(escape(item)) for item in items)


def html_case_rows(rows: List[Tuple[str, str]]) -> str:
    return "".join(
        "<tr><th>{0}</th><td>{1}</td></tr>".format(escape(label), escape(value))
        for label, value in rows
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: #ffffff !important;
            color: #282c3f !important;
        }
        header[data-testid="stHeader"] {
            display: none;
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        #MainMenu, footer {
            visibility: hidden;
        }
        .stApp {
            background: #ffffff;
            color: #282c3f;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] label,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stMarkdownContainer"] h6 {
            color: #282c3f !important;
            opacity: 1 !important;
        }
        .block-container {
            padding-top: 0.35rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }
        .m-nav {
            background: #ffffff;
            border-bottom: 1px solid #eaeaec;
            margin: 0 0 2rem 0;
            padding: 0 0 1.1rem 0;
        }
        .m-nav-inner {
            max-width: 1320px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 1.25rem;
            flex-wrap: wrap;
        }
        .m-logo {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.4rem;
            color: #ffffff;
            background: linear-gradient(135deg, #ff3f6c 0%, #ff6a3d 50%, #a53dff 100%);
        }
        .m-menu {
            display: flex;
            align-items: center;
            gap: 1.6rem;
            font-size: 0.98rem;
            font-weight: 700;
            color: #282c3f;
        }
        .m-menu span {
            white-space: nowrap;
        }
        .m-search {
            flex: 1;
            background: #f5f5f6;
            border-radius: 6px;
            height: 48px;
            display: flex;
            align-items: center;
            padding: 0 1rem;
            color: #94969f;
            font-size: 0.98rem;
        }
        .m-icons {
            display: flex;
            align-items: center;
            gap: 1.3rem;
            color: #282c3f;
            font-weight: 700;
            font-size: 0.92rem;
        }
        .m-icons a {
            color: #282c3f;
            text-decoration: none;
        }
        .m-icons a:hover {
            color: #ff3f6c;
        }
        .wishlist-top {
            padding: 0.05rem 0 0.55rem 0;
        }
        .wishlist-title {
            font-size: 1.7rem;
            font-weight: 800;
            color: #282c3f;
            margin-bottom: 0;
        }
        .wishlist-title span {
            font-weight: 500;
            color: #535766;
        }
        .wishlist-subtitle {
            color: #7e818c;
            font-size: 0.98rem;
        }
        .view-strip {
            background: #ffffff;
            border: 1px solid #f0f1f3;
            border-radius: 16px;
            padding: 0.45rem 0.7rem 0.25rem 0.7rem;
            margin-bottom: 1rem;
        }
        div[data-baseweb="radio"] > div {
            gap: 0.55rem;
        }
        div[role="radiogroup"] label {
            background: #ffffff !important;
            border: 1px solid #e8e9ec !important;
            border-radius: 999px !important;
            padding: 10px 16px !important;
        }
        div[role="radiogroup"] label p,
        div[role="radiogroup"] label span {
            color: #282c3f !important;
            opacity: 1 !important;
        }
        div[role="radiogroup"] label[data-checked="true"] {
            border-color: #ff3f6c !important;
            background: #fff6f8 !important;
        }
        .wishlist-card {
            border: 1px solid #eaeaec;
            border-radius: 4px;
            background: #ffffff;
            margin-bottom: 1.5rem;
        }
        .wishlist-card-shell {
            border: 1px solid #eaeaec;
            border-radius: 4px;
            overflow: hidden;
            background: #ffffff;
            margin-bottom: 1rem;
        }
        .wishlist-card-shell.ready-shell {
            border-color: rgba(6, 95, 70, 1);
            box-shadow: 0 0 0 4px rgba(6, 95, 70, 0.34), 0 18px 36px rgba(6, 95, 70, 0.24);
        }
        .wishlist-card-shell.building-shell {
            border-color: rgba(180, 83, 9, 0.95);
            box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.28), 0 16px 28px rgba(217, 119, 6, 0.18);
        }
        .wish-image-shell {
            position: relative;
            background: #f5f5f6;
            min-height: 260px;
        }
        .wish-image-shell img {
            width: 100%;
            height: 260px;
            object-fit: cover;
            display: block;
        }
        .remove-badge {
            position: absolute;
            top: 14px;
            right: 14px;
            width: 38px;
            height: 38px;
            border-radius: 999px;
            background: rgba(255,255,255,0.92);
            color: #6b7280;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.65rem;
            box-shadow: 0 2px 8px rgba(40,44,63,0.1);
        }
        .verdict-pill {
            position: absolute;
            left: 12px;
            bottom: 12px;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.01em;
        }
        .verdict-pill.ready {
            background: rgba(3, 201, 136, 0.12);
            color: #047857;
        }
        .verdict-pill.building {
            background: rgba(255, 63, 108, 0.12);
            color: #b91c5c;
        }
        .wish-detail {
            padding: 0.8rem 0.85rem 0.65rem 0.85rem;
        }
        .wish-brand {
            font-size: 0.95rem;
            font-weight: 800;
            color: #282c3f;
            margin-bottom: 0.1rem;
        }
        .wish-name {
            color: #535766;
            font-size: 0.92rem;
            line-height: 1.4;
            min-height: 2.45rem;
        }
        .wish-price {
            margin-top: 0.45rem;
            color: #282c3f;
            font-weight: 800;
            font-size: 1rem;
        }
        .wish-price span {
            color: #94969f;
            font-weight: 500;
            text-decoration: line-through;
            margin-left: 0.35rem;
            font-size: 0.86rem;
        }
        .wish-price em {
            color: #ff905a;
            font-style: normal;
            font-weight: 700;
            margin-left: 0.35rem;
            font-size: 0.84rem;
        }
        .wish-note {
            margin-top: 0.45rem;
            color: #7e818c;
            font-size: 0.8rem;
            min-height: 2.2rem;
        }
        .card-actions {
            padding: 0.25rem 1rem 0.6rem 1rem;
        }
        .card-footnote {
            padding: 0 1rem 1rem 1rem;
            color: #7e818c;
            font-size: 0.85rem;
        }
        .section-card {
            background: #ffffff;
            border: 1px solid #ececec;
            border-radius: 16px;
            padding: 0.88rem 0.96rem;
            box-shadow: 0 6px 20px rgba(40, 44, 63, 0.04);
        }
        .verdict-hero {
            background: linear-gradient(135deg, #fff7fa 0%, #ffffff 100%);
            border: 1px solid #f5d5de;
            border-radius: 18px;
            padding: 0.92rem 1.02rem;
            box-shadow: 0 10px 30px rgba(255, 63, 108, 0.08);
        }
        .verdict-product-title {
            color: #282c3f;
            font-size: 1.56rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0.24rem 0 0.28rem 0;
        }
        .verdict-meta-line,
        .verdict-price-line {
            color: #4b5563;
            font-size: 1rem;
            line-height: 1.5;
        }
        .verdict-price-line {
            margin-top: 0.25rem;
            font-weight: 700;
            color: #282c3f;
        }
        .verdict-explainer {
            color: #6b7280;
            font-size: 0.89rem;
            line-height: 1.45;
            margin-top: 0.48rem;
        }
        .decision-card {
            background: #ffffff;
            border: 1px solid #ececec;
            border-radius: 18px;
            padding: 0.92rem 1rem;
            box-shadow: 0 8px 24px rgba(40, 44, 63, 0.05);
        }
        .decision-score {
            color: #282c3f;
            font-size: 1.72rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }
        .decision-label {
            color: #4b5563;
            font-size: 0.94rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .signal-banner {
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            margin-top: 0.85rem;
            font-size: 0.94rem;
            line-height: 1.5;
        }
        .signal-banner.positive {
            background: rgba(6, 95, 70, 0.08);
            border: 1px solid rgba(6, 95, 70, 0.18);
            color: #065f46;
        }
        .signal-banner.caution {
            background: rgba(217, 119, 6, 0.08);
            border: 1px solid rgba(217, 119, 6, 0.18);
            color: #92400e;
        }
        .save-nudge {
            margin-top: 1rem;
            padding: 1rem 1.05rem;
            background: #fff6f8;
            border: 1px solid #ffb5c8;
            border-left: 5px solid #ff3f6c;
            border-radius: 12px;
        }
        .save-nudge h3 {
            margin: 0 0 0.35rem 0;
            color: #282c3f;
            font-size: 1.1rem;
        }
        .save-nudge p {
            margin: 0;
            color: #535766;
            line-height: 1.45;
        }
        .decision-summary {
            margin-top: 1.1rem;
            padding: 1rem 1.05rem;
            border: 1px solid #dce4ea;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 6px 20px rgba(40, 44, 63, 0.04);
        }
        .decision-summary-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 0.8rem;
            margin-top: 0.75rem;
        }
        .decision-summary-item {
            padding: 0.75rem;
            background: #fafbfc;
            border-radius: 10px;
        }
        .decision-summary-label {
            color: #7e818c;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .decision-summary-value {
            margin-top: 0.3rem;
            color: #282c3f;
            font-size: 0.96rem;
            line-height: 1.42;
        }
        .verdict-stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.9rem;
        }
        .verdict-stat {
            border: 1px solid #ececec;
            border-radius: 14px;
            background: #ffffff;
            padding: 0.64rem 0.76rem;
        }
        .verdict-stat-label {
            color: #7e818c;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
        }
        .verdict-stat-value {
            color: #282c3f;
            font-size: 1.02rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .section-card h4,
        .section-card p,
        .section-card li,
        .section-card strong,
        .section-card th,
        .section-card td {
            color: #282c3f;
            opacity: 1;
        }
        .kicker {
            color: #ff3f6c;
            font-weight: 800;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .verdict-copy {
            color: #4b5563;
            line-height: 1.38;
            margin-top: 0.38rem;
            font-size: 0.9rem;
        }
        .evidence-list {
            margin: 0.34rem 0 0 0;
            padding-left: 1rem;
            color: #282c3f;
        }
        .evidence-list li {
            margin-bottom: 0.24rem;
            line-height: 1.32;
            font-size: 0.86rem;
        }
        .case-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.42rem;
        }
        .case-table th,
        .case-table td {
            text-align: left;
            vertical-align: top;
            padding: 0.5rem 0.56rem;
            border-top: 1px solid #ececec;
            font-size: 0.86rem;
        }
        .case-table th {
            width: 122px;
            background: #fafbfc;
            font-weight: 800;
        }
        .metric-card {
            border: 1px solid #ececec;
            border-radius: 14px;
            padding: 1rem;
            background: #fff;
            margin-bottom: 1rem;
        }
        .metric-card .label {
            color: #7e818c;
            font-size: 0.85rem;
        }
        .metric-card .value {
            color: #282c3f;
            font-size: 1.5rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }
        .stButton > button {
            border-radius: 0 !important;
            min-height: 44px !important;
            font-weight: 800 !important;
            border: 1px solid #ff3f6c !important;
            color: #ff3f6c !important;
            background: #ffffff !important;
        }
        .stButton > button[kind="primary"] {
            background: #ff3f6c !important;
            color: #ffffff !important;
        }
        .stButton > button:hover {
            border-color: #ff3f6c !important;
            color: #ff3f6c !important;
            box-shadow: none !important;
        }
        .stButton > button[kind="primary"]:hover {
            color: #ffffff !important;
            background: #ff527b !important;
        }
        [data-testid="stImage"] img {
            border-radius: 0 !important;
        }
        [data-testid="stMetricLabel"] *, 
        [data-testid="stMetricValue"] * {
            color: #282c3f !important;
            opacity: 1 !important;
        }
        .stCaptionContainer, .stCaptionContainer p {
            color: #6b7280 !important;
            opacity: 1 !important;
        }
        .stSelectbox label,
        .stSelectbox p,
        [data-baseweb="select"] *,
        [data-testid="stTable"] *,
        [data-testid="stDataFrame"] *,
        [data-testid="stAlert"] *,
        [data-testid="stExpander"] * {
            color: #282c3f !important;
            opacity: 1 !important;
        }
        [data-baseweb="select"] > div {
            background: #ffffff !important;
            border-color: #d1d5db !important;
        }
        .stSelectbox div[data-baseweb="select"] {
            background: #ffffff !important;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = VIEW_PRODUCT
    if "nav_view" not in st.session_state:
        st.session_state["nav_view"] = st.session_state["active_view"]
    if "selected_product_id" not in st.session_state:
        st.session_state["selected_product_id"] = None
    if "selected_comparison_id" not in st.session_state:
        st.session_state["selected_comparison_id"] = None
    if "last_llm_error" not in st.session_state:
        st.session_state["last_llm_error"] = None
    if "pending_view" not in st.session_state:
        st.session_state["pending_view"] = None
    if "saved_product_ids" not in st.session_state:
        st.session_state["saved_product_ids"] = None
    if "decision_nudge" not in st.session_state:
        st.session_state["decision_nudge"] = None


def open_verdict(product_id: str, hesitation: str, score: int) -> None:
    st.session_state["selected_product_id"] = product_id
    st.session_state["selected_comparison_id"] = None
    st.session_state["pending_view"] = VIEW_VERDICT
    log_event(product_id, "verdict_viewed", {"hesitation": hesitation, "score": score})
    st.rerun()


def product_lookup(products: List[Dict]) -> Dict[str, Dict]:
    return {product["id"]: product for product in products}


def saved_product_ids(products: List[Dict]) -> List[str]:
    if st.session_state["saved_product_ids"] is None:
        st.session_state["saved_product_ids"] = [
            product["id"] for product in products if product["id"] != FEATURED_PRODUCT_ID
        ]
    return st.session_state["saved_product_ids"]


def is_saved(product_id: str, products: List[Dict]) -> bool:
    return product_id in saved_product_ids(products)


def save_with_decision_nudge(product: Dict, products: List[Dict]) -> None:
    saved_ids = saved_product_ids(products)
    if product["id"] not in saved_ids:
        saved_ids.append(product["id"])

    candidate = comparison_target(product, products)
    st.session_state["decision_nudge"] = {
        "product_id": product["id"],
        "candidate_id": candidate["id"] if candidate and candidate["id"] in saved_ids else None,
    }
    log_event(product["id"], "wishlist_saved", {"candidate_id": st.session_state["decision_nudge"]["candidate_id"]})
    if st.session_state["decision_nudge"]["candidate_id"]:
        log_event(product["id"], "decision_nudge_shown", {"candidate_id": st.session_state["decision_nudge"]["candidate_id"]})


def discard_saved_product(product_id: str, source_product_id: str, source: str) -> None:
    st.session_state["saved_product_ids"] = [
        saved_id for saved_id in st.session_state["saved_product_ids"] if saved_id != product_id
    ]
    st.session_state["decision_nudge"] = None
    log_event(source_product_id, "decision_nudge_discarded", {"discarded_product_id": product_id, "source": source})


def comparison_group_for(product: Dict) -> str:
    return str(product.get("comparison_group") or product.get("category") or "").strip()


def comparison_target(product: Dict, products: List[Dict]) -> Optional[Dict]:
    products_by_id = product_lookup(products)
    explicit_id = product.get("comparison_target_id")
    if explicit_id and explicit_id in products_by_id:
        return products_by_id[explicit_id]

    primary_group = comparison_group_for(product)
    category_matches = [
        item for item in products
        if item["id"] != product["id"] and comparison_group_for(item) == primary_group
    ]
    if not category_matches:
        return None

    ranked = sorted(
        category_matches,
        key=lambda item: (
            confidence_score(item, diagnose_hesitation(item["signals"], item)[0]),
            evidence_strength(item),
            -int(item.get("price", 0)),
        ),
        reverse=True,
    )
    return ranked[0]


def risk_markers(product: Dict) -> List[str]:
    joined = " ".join(product.get("reviews", []) + product.get("return_notes", [])).lower()
    markers = []
    for marker in ["tight", "thin", "heavy", "creases", "oversized", "price", "wide feet", "fit mismatch", "narrow"]:
        if marker in joined:
            markers.append(marker)
    return sorted(set(markers))


def build_comparison_resolution(primary: Dict, candidate: Dict) -> Dict[str, object]:
    primary_hesitation, _, primary_score, primary_label, primary_case = get_case_file(primary)
    candidate_hesitation, _, candidate_score, candidate_label, candidate_case = get_case_file(candidate)
    primary_evidence = evidence_strength(primary)
    candidate_evidence = evidence_strength(candidate)
    primary_risks = risk_markers(primary)
    candidate_risks = risk_markers(candidate)

    primary_rank = (primary_score, primary_evidence, -len(primary_risks), -int(primary["price"]))
    candidate_rank = (candidate_score, candidate_evidence, -len(candidate_risks), -int(candidate["price"]))

    winner, winner_hesitation, winner_score, winner_label, winner_case = (
        (primary, primary_hesitation, primary_score, primary_label, primary_case)
        if primary_rank >= candidate_rank
        else (candidate, candidate_hesitation, candidate_score, candidate_label, candidate_case)
    )
    loser, loser_hesitation, loser_score, loser_label, loser_case = (
        (candidate, candidate_hesitation, candidate_score, candidate_label, candidate_case)
        if winner["id"] == primary["id"]
        else (primary, primary_hesitation, primary_score, primary_label, primary_case)
    )

    score_gap = abs(primary_score - candidate_score)
    evidence_gap = abs(primary_evidence - candidate_evidence)
    decisive_edge = []
    if score_gap >= 6:
        decisive_edge.append("higher decision confidence")
    if winner_case["watch_outs"] != loser_case["watch_outs"]:
        decisive_edge.append("fewer buyer risk signals")
    if evidence_gap >= 12:
        decisive_edge.append("deeper evidence maturity")
    if int(winner["price"]) < int(loser["price"]):
        decisive_edge.append("lower spend for the same mission")
    if not decisive_edge:
        decisive_edge.append("clearer buyer proof for this need")

    return {
        "winner": winner,
        "loser": loser,
        "winner_hesitation": winner_hesitation,
        "winner_score": winner_score,
        "winner_label": winner_label,
        "winner_case": winner_case,
        "loser_hesitation": loser_hesitation,
        "loser_score": loser_score,
        "loser_label": loser_label,
        "loser_case": loser_case,
        "primary_hesitation": primary_hesitation,
        "candidate_hesitation": candidate_hesitation,
        "primary_score": primary_score,
        "candidate_score": candidate_score,
        "primary_evidence": primary_evidence,
        "candidate_evidence": candidate_evidence,
        "primary_risks": primary_risks,
        "candidate_risks": candidate_risks,
        "winner_risks": primary_risks if winner["id"] == primary["id"] else candidate_risks,
        "loser_risks": candidate_risks if winner["id"] == primary["id"] else primary_risks,
        "decisive_edge": decisive_edge,
    }


def open_comparison(primary_id: str, candidate_id: Optional[str]) -> None:
    st.session_state["selected_product_id"] = primary_id
    st.session_state["selected_comparison_id"] = candidate_id
    st.session_state["pending_view"] = VIEW_COMPARISON
    log_event(primary_id, "comparison_opened", {"candidate_id": candidate_id})
    st.rerun()


@st.dialog("Decision Nudge")
def render_decision_nudge_dialog(product: Dict, candidate: Dict) -> None:
    st.markdown(
        """
        <div class='save-nudge'>
          <h3>You already saved a similar shoe</h3>
          <p><strong>{candidate}</strong> is already in your wishlist. Compare the two now to make a choice, or remove the alternative.</p>
        </div>
        """.format(candidate=escape(candidate["name"])),
        unsafe_allow_html=True,
    )
    st.caption("This nudge appears at the moment of saving, before the decision is deferred.")

    if st.button("Compare saved shoes", key="nudge_compare", type="primary", use_container_width=True):
        log_event(product["id"], "decision_nudge_comparison_opened", {"candidate_id": candidate["id"]})
        open_comparison(product["id"], candidate["id"])
    if st.button("Remove {0}".format(short_title(candidate["name"], 24)), key="nudge_discard", use_container_width=True):
        discard_saved_product(candidate["id"], product["id"], "save_nudge")
        st.rerun()
    if st.button("Keep both for now", key="nudge_keep_both", use_container_width=True):
        st.session_state["decision_nudge"] = None
        log_event(product["id"], "decision_nudge_deferred", {"candidate_id": candidate["id"]})
        st.rerun()


def render_top_nav() -> None:
    st.markdown(
        """
        <div class="m-nav">
          <div class="m-nav-inner">
            <div class="m-logo">M</div>
            <div class="m-menu">
              <span>MEN</span>
              <span>WOMEN</span>
              <span>KIDS</span>
              <span>HOME</span>
              <span>BEAUTY</span>
              <span>GENZ</span>
              <span>STUDIO</span>
            </div>
            <div class="m-search">🔎&nbsp;&nbsp;Search for products, brands and more</div>
            <div class="m-icons">
              <span>Profile</span>
              <a href="?view=wishlist" target="_self" aria-label="Open wishlist">Wishlist</a>
              <span>Bag</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(products: List[Dict]) -> None:
    current_view = st.session_state.get("active_view")
    render_top_nav()
    if current_view == VIEW_WISHLIST:
        st.markdown(
            "<div class='wishlist-top'><div class='wishlist-title'>My Wishlist <span>{0} items</span></div></div>".format(
                len(products)
            ),
            unsafe_allow_html=True,
        )
    if st.session_state.get("last_llm_error"):
        st.warning("Groq fallback mode is active because the LLM request failed: {0}".format(st.session_state["last_llm_error"]))


def apply_requested_view() -> None:
    requested_view = st.query_params.get("view")
    if requested_view == "wishlist":
        st.session_state["active_view"] = VIEW_WISHLIST
        st.session_state["nav_view"] = VIEW_WISHLIST
        st.query_params.clear()


def render_navigation() -> None:
    if st.session_state.get("active_view") in {VIEW_VERDICT, VIEW_COMPARISON}:
        return
    visible_views = [VIEW_PRODUCT, VIEW_WISHLIST, VIEW_VERDICT, VIEW_ANALYTICS]
    if st.session_state.get("active_view") == VIEW_COMPARISON:
        st.session_state["nav_view"] = VIEW_VERDICT
    elif st.session_state.get("nav_view") not in visible_views:
        st.session_state["nav_view"] = st.session_state.get("active_view", VIEW_WISHLIST)

    st.markdown("<div class='view-strip'></div>", unsafe_allow_html=True)
    selected_view = st.radio(
        "View",
        visible_views,
        horizontal=True,
        label_visibility="collapsed",
        key="nav_view",
    )
    if st.session_state.get("active_view") == VIEW_COMPARISON and selected_view == VIEW_VERDICT:
        return
    if selected_view != st.session_state.get("active_view"):
        st.session_state["active_view"] = selected_view


def render_product_card(product: Dict) -> None:
    status = maturity_status(product)
    hesitation, _, score, _, _ = get_case_file(product)
    price, mrp, discount = pricing_snapshot(product)
    shell_class = "ready-shell" if status == "Verdict Ready" else "building-shell"
    image = image_src(product["image"])
    with st.container():
        st.markdown(
            """
            <div class="wishlist-card-shell {shell_class}">
              <div class="wish-image-shell">
                <img src="{image}" alt="{name}" />
                <div class="remove-badge">×</div>
              </div>
              <div class="wish-detail">
                <div class="wish-brand">{brand}</div>
                <div class="wish-name">{name}</div>
                <div class="wish-price">₹{price}<span>₹{mrp}</span><em>({discount}% OFF)</em></div>
                <div class="wish-note">{hesitation} · Confidence {score}% · {age}h in wishlist</div>
              </div>
            </div>
            """.format(
                image=image,
                name=short_title(product["name"], 40),
                brand=product["brand"],
                price=f"{price:,}",
                mrp=f"{mrp:,}",
                discount=discount,
                hesitation=hesitation,
                score=score,
                age=product["wishlist_age_hours"],
                shell_class=shell_class,
            ),
            unsafe_allow_html=True,
        )
        if status == "Verdict Ready":
            action_left, action_right = st.columns(2)
            with action_left:
                if st.button("MOVE TO BAG", key="bag_{0}".format(product["id"]), use_container_width=True):
                    log_event(product["id"], "move_to_bag_clicked", {"status": status, "score": score})
                    st.success("Move-to-bag interaction captured.")
            with action_right:
                if st.button("OPEN VERDICT", key="open_{0}".format(product["id"]), type="primary", use_container_width=True):
                    open_verdict(product["id"], hesitation, score)
        else:
            if st.button("MOVE TO BAG", key="bag_{0}".format(product["id"]), use_container_width=True):
                log_event(product["id"], "move_to_bag_clicked", {"status": status, "score": score})
                st.success("Move-to-bag interaction captured.")
def render_wishlist(products: List[Dict]) -> None:
    for row_start in range(0, len(products), 4):
        columns = st.columns(4)
        for offset, product in enumerate(products[row_start : row_start + 4]):
            with columns[offset]:
                render_product_card(product)


def render_product_page(product: Dict, products: List[Dict]) -> None:
    price, mrp, discount = pricing_snapshot(product)
    st.markdown("<div class='kicker'>Product page</div>", unsafe_allow_html=True)
    product_left, product_right = st.columns([1, 1.15])
    with product_left:
        st.image(product["image"], use_column_width=True)
    with product_right:
        st.markdown(
            """
            <div class='section-card'>
              <div class='kicker'>{brand}</div>
              <h2 style='margin:0.32rem 0 0.22rem 0;'>{name}</h2>
              <div class='verdict-copy'>{category} · Everyday casual sneaker</div>
              <div class='verdict-price-line'>₹{price} <span style='color:#94969f;text-decoration:line-through;font-weight:500;'>₹{mrp}</span> <span style='color:#ff905a;font-size:0.9rem;'>{discount}% OFF</span></div>
              <div class='verdict-copy'><strong>Why it is worth considering:</strong> Comfortable for everyday wear with strong review coverage, but shoppers often compare it with similar white sneakers before deciding.</div>
            </div>
            """.format(
                brand=escape(product["brand"]),
                name=escape(product["name"]),
                category=escape(product["category"]),
                price=f"{price:,}",
                mrp=f"{mrp:,}",
                discount=discount,
            ),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top:0.72rem;'><strong>Select size</strong></div>", unsafe_allow_html=True)
        st.radio("Size", ["7", "8", "9", "10"], horizontal=True, index=2, label_visibility="collapsed", key="featured_size")
        if is_saved(product["id"], products):
            st.success("Saved to Wishlist")
            if st.button("Review similar saved shoes", key="review_featured_product", use_container_width=True):
                candidate = comparison_target(product, products)
                st.session_state["decision_nudge"] = {
                    "product_id": product["id"],
                    "candidate_id": candidate["id"] if candidate and is_saved(candidate["id"], products) else None,
                }
                st.rerun()
        elif st.button("Save to Wishlist", key="save_featured_product", type="primary", use_container_width=True):
            save_with_decision_nudge(product, products)
            st.rerun()

    nudge = st.session_state.get("decision_nudge")
    if nudge and nudge.get("product_id") == product["id"]:
        candidate = product_lookup(products).get(nudge.get("candidate_id"))
        if candidate and is_saved(candidate["id"], products):
            render_decision_nudge_dialog(product, candidate)


def render_comparison_decision_summary(primary: Dict, candidate: Dict, resolution: Dict[str, object]) -> None:
    winner = resolution["winner"]
    loser = resolution["loser"]
    primary_hesitation, primary_explanation = diagnose_hesitation(primary["signals"], primary)
    primary_signals = primary["signals"]
    st.markdown(
        """
        <div class='decision-summary'>
          <div class='kicker'>Decision summary</div>
          <div class='verdict-copy'><strong>Recommended choice:</strong> {winner}</div>
          <div class='verdict-copy'><strong>Why it is the stronger choice:</strong> {why}</div>
          <div class='decision-summary-grid'>
            <div class='decision-summary-item'>
              <div class='decision-summary-label'>Your hesitation</div>
              <div class='decision-summary-value'><strong>{hesitation}</strong><br>{explanation}</div>
            </div>
            <div class='decision-summary-item'>
              <div class='decision-summary-label'>Your decision signals</div>
              <div class='decision-summary-value'>{size_chart} seconds on size chart · {review_scroll}% review scroll<br>{similar} similar products viewed · {revisits} repeat visits</div>
            </div>
            <div class='decision-summary-item'>
              <div class='decision-summary-label'>Evidence behind the choice</div>
              <div class='decision-summary-value'>{reviews} reviews · {buyer_notes} similar-buyer signals · {returns} return-data signals<br>{risks}</div>
            </div>
          </div>
          <div class='verdict-copy'><strong>Next action:</strong> Add {winner} to bag. Keep {loser} only if you still need a backup option.</div>
        </div>
        """.format(
            winner=escape(winner["name"]),
            loser=escape(loser["name"]),
            why=escape("It has {0} with {1}% confidence versus {2}% on the alternative.".format(", ".join(resolution["decisive_edge"]), resolution["winner_score"], resolution["loser_score"])),
            hesitation=escape(primary_hesitation),
            explanation=escape(primary_explanation),
            size_chart=primary_signals.get("size_chart_seconds", 0),
            review_scroll=primary_signals.get("review_scroll_percent", 0),
            similar=primary_signals.get("similar_products_viewed", 0),
            revisits=primary_signals.get("pdp_revisits", 0),
            reviews=len(winner.get("reviews", [])),
            buyer_notes=len(winner.get("similar_buyer_notes", [])),
            returns=len(winner.get("return_notes", [])),
            risks=escape("Risk flags: {0}".format(", ".join(resolution["winner_risks"]) if resolution["winner_risks"] else "low observed risk")),
        ),
        unsafe_allow_html=True,
    )

    action_left, action_mid, action_right = st.columns(3)
    with action_left:
        if st.button("Add {0} to Bag".format(short_title(winner["name"], 18)), key="add_winner_to_bag", type="primary", use_container_width=True):
            log_event(winner["id"], "comparison_add_to_bag", {"loser_id": loser["id"], "winner_score": resolution["winner_score"]})
            st.success("{0} was added to bag. {1} remains saved as your alternative.".format(winner["name"], loser["name"]))
    with action_mid:
        if st.button("Buy {0} now".format(short_title(winner["name"], 18)), key="buy_winner_now", use_container_width=True):
            log_event(winner["id"], "comparison_buy_now", {"loser_id": loser["id"], "winner_score": resolution["winner_score"]})
            st.success("Buy-now flow started for {0}.".format(winner["name"]))
    with action_right:
        if st.button("Keep both saved", key="keep_both", use_container_width=True):
            log_event(primary["id"], "comparison_deferred", {"candidate_id": candidate["id"]})
            st.info("Deferred. In production, the app would remind the shopper only after new evidence arrives.")


def render_verdict(product: Dict) -> None:
    hesitation, explanation, score, label, case = get_case_file(product)
    price, mrp, discount = pricing_snapshot(product)
    positive_signals = case.get("positive_signals", "Positive signals are still forming.")
    signal_class = "positive" if score >= 78 else "caution"
    evidence_total = evidence_strength(product)

    header_left, header_right = st.columns([1.8, 1])
    with header_left:
        st.markdown(
            """
            <div class='verdict-hero'>
              <div class='kicker'>Verdict card</div>
              <div class='verdict-product-title'>{name}</div>
              <div class='verdict-meta-line'><strong>{brand}</strong> · {category}</div>
              <div class='verdict-price-line'>₹{price} · MRP ₹{mrp} · {discount}% OFF</div>
              <div class='verdict-explainer'>{explanation}</div>
            </div>
            """.format(
                name=escape(product["name"]),
                brand=escape(product["brand"]),
                category=escape(product["category"]),
                price=f"{price:,}",
                mrp=f"{mrp:,}",
                discount=discount,
                explanation=escape(explanation),
            ),
            unsafe_allow_html=True,
        )
    with header_right:
        if st.button("← Back to Wishlist", use_container_width=True):
            st.session_state["pending_view"] = VIEW_WISHLIST
            st.rerun()
        st.markdown(
            "<div class='decision-card'><div class='kicker'>Decision confidence</div><div class='decision-score'>{0}%</div><div class='decision-label'>{1}</div></div>".format(
                score, escape(label)
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class='section-card'>
          <div class='kicker'>Decision now</div>
          <div class='verdict-copy'><strong>Recommended action:</strong> {recommended_action}</div>
          <div class='verdict-copy'>{positive_signals}</div>
          <div class='verdict-copy'>{watch_outs}</div>
        </div>
        """.format(
            recommended_action=escape(case["recommended_action"]),
            positive_signals=escape(positive_signals),
            watch_outs=escape(case["watch_outs"]),
        ),
        unsafe_allow_html=True,
    )
    top_action_left, top_action_mid, top_action_right = st.columns(3)
    with top_action_left:
        if st.button("Add to Cart", key="verdict_add_to_cart", type="primary", use_container_width=True):
            log_event(product["id"], "add_to_cart", {"source": "verdict_card", "score": score})
            st.success("Added to cart based on the verdict.")
    with top_action_mid:
        if st.button("Buy Now", key="verdict_buy_now", use_container_width=True):
            log_event(product["id"], "buy_now_clicked", {"source": "verdict_card", "score": score})
            st.success("Buy-now flow started for this item.")
    with top_action_right:
        if st.button("Open Comparison Analysis", key="comparison_top", use_container_width=True):
            candidate = comparison_target(product, load_products())
            log_event(product["id"], "compare_again", {"similar_items": product.get("similar_saved_items", []), "candidate_id": candidate.get("id") if candidate else None})
            if candidate:
                open_comparison(product["id"], candidate["id"])
            else:
                st.info("No comparison candidate is available yet for this product.")

    content_left, content_mid, content_right = st.columns([0.9, 1.18, 0.92])
    with content_left:
        st.image(product["image"], use_column_width=True)
        st.progress(score / 100, text="Confidence score: {0}%".format(score))
        st.markdown(
            """
            <div class='verdict-stats'>
              <div class='verdict-stat'>
                <div class='verdict-stat-label'>Reviews</div>
                <div class='verdict-stat-value'>{reviews}</div>
              </div>
              <div class='verdict-stat'>
                <div class='verdict-stat-label'>Return Signals</div>
                <div class='verdict-stat-value'>{returns}</div>
              </div>
              <div class='verdict-stat'>
                <div class='verdict-stat-label'>Comparisons</div>
                <div class='verdict-stat-value'>{comparisons}</div>
              </div>
              <div class='verdict-stat'>
                <div class='verdict-stat-label'>Evidence</div>
                <div class='verdict-stat-value'>{evidence}</div>
              </div>
            </div>
            """.format(
                reviews=len(product.get("reviews", [])),
                returns=len(product.get("return_notes", [])),
                comparisons=len(product.get("similar_saved_items", [])),
                evidence=evidence_total,
            ),
            unsafe_allow_html=True,
        )
    with content_mid:
        rows = [
            ("Fit", case["fit_summary"]),
            ("Positive signals", case.get("positive_signals", "Positive signals are still forming.")),
            ("Reviews", case["review_summary"]),
            ("Styling / Occasion", case["styling_use_case"]),
            ("Comparison", case["comparison_note"]),
            ("Watch-outs", case["watch_outs"]),
            ("Recommended action", case["recommended_action"]),
        ]
        st.markdown(
            """
            <div class='section-card'>
              <div class='kicker'>Case file</div>
              <table class='case-table'>
                {rows}
              </table>
            </div>
            """.format(rows=html_case_rows(rows)),
            unsafe_allow_html=True,
        )
    with content_right:
        st.markdown(
            """
            <div class='section-card'>
              <div class='kicker'>Evidence snapshot</div>
              <div class='verdict-copy'><strong>Primary hesitation:</strong> {hesitation}</div>
              <div class='signal-banner {signal_class}'>{positive_signals}</div>
              <div class='kicker' style='margin-top:0.8rem;'>Saved shortlist</div>
              <ul class='evidence-list'>{similar_items}</ul>
              <div class='kicker' style='margin-top:0.8rem;'>Buyer evidence</div>
              <ul class='evidence-list'>{buyer_notes}</ul>
              <div class='kicker' style='margin-top:0.8rem;'>Review highlights</div>
              <ul class='evidence-list'>{reviews}</ul>
              <div class='kicker' style='margin-top:0.8rem;'>Return &amp; trust signals</div>
              <ul class='evidence-list'>{notes}</ul>
            </div>
            """.format(
                hesitation=escape(hesitation),
                positive_signals=escape(positive_signals),
                signal_class=signal_class,
                similar_items=html_bullets(product.get("similar_saved_items", [])),
                buyer_notes=html_bullets(product.get("similar_buyer_notes", [])),
                reviews=html_bullets(product.get("reviews", [])[:3]),
                notes=html_bullets(product.get("return_notes", [])[:3]),
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Still disagree with this verdict?"):
        reason = st.selectbox(
            "What still feels unresolved?",
            [
                "Size still unclear",
                "Don't trust reviews",
                "Found better option elsewhere",
                "Price Doubt / value unclear",
                "Not needed anymore",
                "Style does not feel right",
            ],
            key="override_reason_{0}".format(product["id"]),
        )
        if st.button("Override Verdict", key="override_{0}".format(product["id"])):
            log_event(product["id"], "override_verdict", {"reason": reason, "score": score, "hesitation": hesitation})
            st.warning("Override captured. This would become a training signal for future verdict quality.")


def render_comparison(products: List[Dict]) -> None:
    selected_id = st.session_state.get("selected_product_id")
    selected_comparison_id = st.session_state.get("selected_comparison_id")
    products_by_id = product_lookup(products)
    primary = products_by_id.get(selected_id)

    if not primary:
        st.info("Open a verdict card first to compare against another saved product.")
        if st.button("Go to Wishlist", key="comparison_to_wishlist"):
            st.session_state["pending_view"] = VIEW_WISHLIST
            st.rerun()
        return

    candidate = products_by_id.get(selected_comparison_id) if selected_comparison_id else comparison_target(primary, products)
    if not candidate:
        st.info("This saved product does not have a strong comparison candidate yet.")
        if st.button("Back to Verdict", key="comparison_to_verdict"):
            st.session_state["pending_view"] = VIEW_VERDICT
            st.rerun()
        return

    resolution = build_comparison_resolution(primary, candidate)
    winner = resolution["winner"]
    loser = resolution["loser"]

    top_left, top_right = st.columns([1.6, 1])
    with top_left:
        st.markdown("<div class='kicker'>Comparison analysis</div>", unsafe_allow_html=True)
        st.title("Resolve comparison paralysis")
        st.write("Side-by-side decision support for two saved products in the same mission.")
        st.caption("Comparisons are only resolved within the same decision mission, not across unrelated wishlist categories.")
        st.caption("This simulates the intended MVP flow: detect indecision, compare evidence, and recommend one next action.")
    with top_right:
        if st.button("← Back to Verdict", key="comparison_back_verdict", use_container_width=True):
            st.session_state["pending_view"] = VIEW_VERDICT
            st.rerun()
        st.markdown(
            "<div class='section-card'><div class='kicker'>Recommended now</div><h3 style='margin:8px 0 6px 0;'>{0}</h3><div style='font-size:1rem;color:#6b7280;'>{1}</div></div>".format(
                escape(winner["name"]),
                escape(", ".join(resolution["decisive_edge"]))
            ),
            unsafe_allow_html=True,
        )

    render_comparison_decision_summary(primary, candidate, resolution)

    compare_left, compare_right = st.columns(2)
    for column, product, hesitation, score, evidence, risks in [
        (compare_left, primary, resolution["primary_hesitation"], resolution["primary_score"], resolution["primary_evidence"], resolution["primary_risks"]),
        (compare_right, candidate, resolution["candidate_hesitation"], resolution["candidate_score"], resolution["candidate_evidence"], resolution["candidate_risks"]),
    ]:
        price, mrp, discount = pricing_snapshot(product)
        with column:
            st.image(product["image"], use_column_width=True)
            st.markdown(
                """
                <div class='section-card'>
                  <div class='kicker'>{status}</div>
                  <h3 style='margin:0.35rem 0 0.25rem 0;'>{name}</h3>
                  <div class='verdict-copy'><strong>{brand}</strong> · {category}</div>
                  <div class='verdict-copy'>₹{price} · MRP ₹{mrp} · {discount}% OFF</div>
                  <div class='verdict-copy'><strong>Hesitation:</strong> {hesitation}</div>
                  <div class='verdict-copy'><strong>Confidence:</strong> {score}%</div>
                  <div class='verdict-copy'><strong>Evidence maturity:</strong> {evidence}</div>
                  <div class='verdict-copy'><strong>Risk flags:</strong> {risks}</div>
                </div>
                """.format(
                    status=escape(maturity_status(product)),
                    name=escape(product["name"]),
                    brand=escape(product["brand"]),
                    category=escape(product["category"]),
                    price=f"{price:,}",
                    mrp=f"{mrp:,}",
                    discount=discount,
                    hesitation=escape(hesitation),
                    score=score,
                    evidence=evidence,
                    risks=escape(", ".join(risks) if risks else "Low observed risk"),
                ),
                unsafe_allow_html=True,
            )

def render_analytics() -> None:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>MVP analytics</div>", unsafe_allow_html=True)
    st.subheader("Interaction log")
    df = get_events()
    if df.empty:
        st.info("No events yet. Open verdict cards and click actions to generate prototype analytics.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    metric_left, metric_mid_left, metric_mid_right, metric_right = st.columns(4)
    metric_left.metric("Verdict views", int((df["event_name"] == "verdict_viewed").sum()))
    metric_mid_left.metric("Buy Now clicks", int((df["event_name"] == "buy_now_clicked").sum()))
    metric_mid_right.metric("Add-to-bag", int((df["event_name"] == "add_to_bag").sum()))
    metric_right.metric("Overrides", int((df["event_name"] == "override_verdict").sum()))

    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_pm_notes() -> None:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>PM framing</div>", unsafe_allow_html=True)
    st.markdown(
        """
### Hypothesis tested
High-intent wishlist users fail to purchase because the wishlist captures the product but does not resolve the hesitation behind the save.

### MVP mechanism
Diagnose hesitation → Build Case File → Surface Verdict Card → Capture Trust / Override → Learn from feedback.

### Why this is non-monetary
The prototype does not use discounts, coupons, or price locks. It attempts to improve conversion by increasing purchase confidence.

### Success metrics to show in the deck
- Verdict Card open rate
- Buy Now rate
- Wishlist-to-bag rate after verdict
- Override rate and override reasons
- Return / size-return guardrail
- User trust feedback

### Production version would require
- Real clickstream events
- Real review and return data
- Size and fit models
- Personalization safeguards
- A/B testing against static wishlist

### Groq integration
Set `GROQ_API_KEY` in `.env` to generate Case Files using Groq. Default model: `llama-3.3-70b-versatile`. If no key is present, the app falls back to deterministic local summaries.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    init_db()
    init_state()
    apply_theme()
    apply_requested_view()
    if st.session_state.get("pending_view"):
        st.session_state["active_view"] = st.session_state["pending_view"]
        st.session_state["nav_view"] = (
            VIEW_VERDICT if st.session_state["pending_view"] == VIEW_COMPARISON else st.session_state["pending_view"]
        )
        st.session_state["pending_view"] = None
    products = load_products()
    products_by_id = product_lookup(products)
    featured_product = products_by_id.get(FEATURED_PRODUCT_ID)
    wishlist_products = [
        product for product in products if product["id"] in saved_product_ids(products)
    ]

    render_header(wishlist_products)
    render_navigation()

    selected_id = st.session_state.get("selected_product_id")
    selected_product = next((product for product in products if product["id"] == selected_id), None)

    if st.session_state["active_view"] == VIEW_PRODUCT:
        if featured_product:
            render_product_page(featured_product, products)
        else:
            st.error("The featured sneaker could not be loaded.")
    elif st.session_state["active_view"] == VIEW_WISHLIST:
        render_wishlist(wishlist_products)
    elif st.session_state["active_view"] == VIEW_VERDICT:
        if not selected_product:
            st.info("Open a verdict card from the wishlist to see the detailed decision view.")
            if st.button("Go to Wishlist"):
                st.session_state["active_view"] = VIEW_WISHLIST
                st.rerun()
        else:
            render_verdict(selected_product)
    elif st.session_state["active_view"] == VIEW_COMPARISON:
        render_comparison(products)
    elif st.session_state["active_view"] == VIEW_ANALYTICS:
        render_analytics()


if __name__ == "__main__":
    main()
