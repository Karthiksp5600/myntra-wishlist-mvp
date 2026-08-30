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
VIEW_VERDICT = "Verdict Card"
VIEW_ANALYTICS = "Analytics"
VIEW_PM = "PM Notes"

load_dotenv()

st.set_page_config(
    page_title="Myntra Wishlist Verdict MVP",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
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


def maturity_status(product: Dict) -> str:
    signals = product["signals"]
    evidence_strength = (
        len(product.get("reviews", [])) * 8
        + len(product.get("return_notes", [])) * 8
        + len(product.get("similar_buyer_notes", [])) * 10
        + min(product.get("wishlist_age_hours", 0), 72) * 0.6
        + signals.get("review_scroll_percent", 0) * 0.2
    )
    return "Verdict Ready" if evidence_strength >= 110 else "Building Verdict"


def confidence_score(product: Dict, hesitation: str) -> int:
    base = 58
    review_count = len(product.get("reviews", []))
    buyer_notes = len(product.get("similar_buyer_notes", []))
    age = min(product.get("wishlist_age_hours", 0), 72)

    score = base + review_count * 4 + buyer_notes * 5 + age * 0.15

    notes = " ".join(product.get("return_notes", [])).lower()
    if "high returns" in notes or "higher" in notes:
        score -= 8
    if "low return" in notes or "low size return" in notes:
        score += 6
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
        return "Trust the verdict and add to bag if the shopper still wants this category."
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
fit_summary, review_summary, styling_use_case, comparison_note, watch_outs, recommended_action.
Keep each value under 35 words.
Do not invent discounts. Do not use urgency tricks.
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
            padding-top: 1rem;
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
        .wishlist-top {
            padding: 0.1rem 0 1rem 0;
        }
        .wishlist-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #282c3f;
            margin-bottom: 0.35rem;
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
            padding: 0.7rem 0.85rem 0.45rem 0.85rem;
            margin-bottom: 1.4rem;
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
            margin-bottom: 1.5rem;
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
            min-height: 320px;
        }
        .wish-image-shell img {
            width: 100%;
            height: 320px;
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
            padding: 1rem 1rem 0.75rem 1rem;
        }
        .wish-brand {
            font-size: 1.05rem;
            font-weight: 800;
            color: #282c3f;
            margin-bottom: 0.1rem;
        }
        .wish-name {
            color: #535766;
            font-size: 1rem;
            line-height: 1.4;
            min-height: 2.8rem;
        }
        .wish-price {
            margin-top: 0.65rem;
            color: #282c3f;
            font-weight: 800;
            font-size: 1.12rem;
        }
        .wish-price span {
            color: #94969f;
            font-weight: 500;
            text-decoration: line-through;
            margin-left: 0.35rem;
            font-size: 0.98rem;
        }
        .wish-price em {
            color: #ff905a;
            font-style: normal;
            font-weight: 700;
            margin-left: 0.35rem;
            font-size: 0.98rem;
        }
        .wish-note {
            margin-top: 0.45rem;
            color: #7e818c;
            font-size: 0.88rem;
            min-height: 2.6rem;
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
            padding: 1.15rem 1.2rem;
            box-shadow: 0 6px 20px rgba(40, 44, 63, 0.04);
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
            line-height: 1.55;
            margin-top: 0.65rem;
        }
        .evidence-list {
            margin: 0.6rem 0 0 0;
            padding-left: 1rem;
            color: #282c3f;
        }
        .evidence-list li {
            margin-bottom: 0.45rem;
            line-height: 1.5;
        }
        .case-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.7rem;
        }
        .case-table th,
        .case-table td {
            text-align: left;
            vertical-align: top;
            padding: 0.8rem 0.75rem;
            border-top: 1px solid #ececec;
            font-size: 0.95rem;
        }
        .case-table th {
            width: 180px;
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
        st.session_state["active_view"] = VIEW_WISHLIST
    if "nav_view" not in st.session_state:
        st.session_state["nav_view"] = st.session_state["active_view"]
    if "selected_product_id" not in st.session_state:
        st.session_state["selected_product_id"] = None
    if "last_llm_error" not in st.session_state:
        st.session_state["last_llm_error"] = None
    if "pending_view" not in st.session_state:
        st.session_state["pending_view"] = None


def open_verdict(product_id: str, hesitation: str, score: int) -> None:
    st.session_state["selected_product_id"] = product_id
    st.session_state["pending_view"] = VIEW_VERDICT
    log_event(product_id, "verdict_viewed", {"hesitation": hesitation, "score": score})
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
              <span>Wishlist</span>
              <span>Bag</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(products: List[Dict]) -> None:
    ready_count = sum(1 for product in products if maturity_status(product) == "Verdict Ready")
    avg_confidence = int(round(sum(get_case_file(product)[2] for product in products) / max(len(products), 1)))

    render_top_nav()
    st.markdown(
        "<div class='wishlist-top'><div class='wishlist-title'>My Wishlist <span>{0} items</span></div><div class='wishlist-subtitle'>Shop the saved list like Myntra, then open a full-page verdict only when you want the decision layer.</div></div>".format(
            len(products)
        ),
        unsafe_allow_html=True,
    )

    meta_left, meta_mid, meta_right = st.columns(3)
    with meta_left:
        st.markdown(
            "<div class='metric-card'><div class='label'>Verdict-ready now</div><div class='value'>{0}</div></div>".format(
                ready_count
            ),
            unsafe_allow_html=True,
        )
    with meta_mid:
        st.markdown(
            "<div class='metric-card'><div class='label'>Needs more confidence</div><div class='value'>{0}</div></div>".format(
                len(products) - ready_count
            ),
            unsafe_allow_html=True,
        )
    with meta_right:
        st.markdown(
            "<div class='metric-card'><div class='label'>Average confidence</div><div class='value'>{0}%</div></div>".format(
                avg_confidence
            ),
            unsafe_allow_html=True,
        )
    if st.session_state.get("last_llm_error"):
        st.warning("Groq fallback mode is active because the LLM request failed: {0}".format(st.session_state["last_llm_error"]))


def render_navigation() -> None:
    st.markdown("<div class='view-strip'></div>", unsafe_allow_html=True)
    selected_view = st.radio(
        "View",
        [VIEW_WISHLIST, VIEW_VERDICT, VIEW_ANALYTICS, VIEW_PM],
        horizontal=True,
        label_visibility="collapsed",
        key="nav_view",
    )
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
        st.caption(
            "Verdict opens now with full evidence."
            if status == "Verdict Ready"
            else "Verdict is unavailable until enough confidence is reached."
        )


def render_wishlist(products: List[Dict]) -> None:
    for row_start in range(0, len(products), 4):
        columns = st.columns(4)
        for offset, product in enumerate(products[row_start : row_start + 4]):
            with columns[offset]:
                render_product_card(product)


def render_verdict(product: Dict) -> None:
    hesitation, explanation, score, label, case = get_case_file(product)
    price, mrp, discount = pricing_snapshot(product)

    header_left, header_right = st.columns([1.8, 1])
    with header_left:
        st.markdown("<div class='kicker'>Verdict card</div>", unsafe_allow_html=True)
        st.title(product["name"])
        st.write("**{0}** · {1}".format(product["brand"], product["category"]))
        st.write("₹{0}  ·  MRP ₹{1}  ·  {2}% OFF".format(f"{price:,}", f"{mrp:,}", discount))
        st.caption(explanation)
    with header_right:
        if st.button("← Back to Wishlist", use_container_width=True):
            st.session_state["pending_view"] = VIEW_WISHLIST
            st.rerun()
        st.markdown(
            "<div class='section-card'><div class='kicker'>Decision confidence</div><h3 style='margin:8px 0 6px 0;'>{0}</h3><div style='font-size:1.75rem;font-weight:800;color:#282c3f;'>{1}%</div></div>".format(
                label, score
            ),
            unsafe_allow_html=True,
        )

    hero_left, hero_right = st.columns([1, 1.25])
    with hero_left:
        st.image(product["image"], use_column_width=True)
        st.progress(score / 100, text="Confidence score: {0}%".format(score))
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Reviews", len(product.get("reviews", [])))
        metric_col2.metric("Return signals", len(product.get("return_notes", [])))
        metric_col3.metric("Comparisons", len(product.get("similar_saved_items", [])))
    with hero_right:
        st.markdown(
            """
            <div class='section-card'>
              <div class='kicker'>Why this verdict helps</div>
              <div class='verdict-copy'><strong>Primary hesitation:</strong> {hesitation}</div>
              <div class='verdict-copy'><strong>Recommended action:</strong> {recommended_action}</div>
              <div class='verdict-copy'><strong>Watch-outs:</strong> {watch_outs}</div>
            </div>
            """.format(
                hesitation=escape(hesitation),
                recommended_action=escape(case["recommended_action"]),
                watch_outs=escape(case["watch_outs"]),
            ),
            unsafe_allow_html=True,
        )

    case_left, case_right = st.columns([1.15, 0.85])
    with case_left:
        rows = [
            ("Fit", case["fit_summary"]),
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
    with case_right:
        st.markdown(
            """
            <div class='section-card'>
              <div class='kicker'>Similar saved items</div>
              <ul class='evidence-list'>{similar_items}</ul>
              <div class='kicker' style='margin-top:1rem;'>Buyer evidence</div>
              <ul class='evidence-list'>{buyer_notes}</ul>
            </div>
            """.format(
                similar_items=html_bullets(product.get("similar_saved_items", [])),
                buyer_notes=html_bullets(product.get("similar_buyer_notes", [])),
            ),
            unsafe_allow_html=True,
        )

    reviews_col, returns_col = st.columns(2)
    with reviews_col:
        st.markdown(
            """
            <div class='section-card'>
              <div class='kicker'>Review highlights</div>
              <ul class='evidence-list'>{reviews}</ul>
            </div>
            """.format(reviews=html_bullets(product.get("reviews", []))),
            unsafe_allow_html=True,
        )
    with returns_col:
        st.markdown(
            """
            <div class='section-card'>
              <div class='kicker'>Return &amp; trust signals</div>
              <ul class='evidence-list'>{notes}</ul>
            </div>
            """.format(notes=html_bullets(product.get("return_notes", []))),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='kicker' style='margin-top:0.4rem;'>Decision</div>", unsafe_allow_html=True)
    st.markdown("#### Decision")
    decision_left, decision_mid, decision_right = st.columns(3)
    with decision_left:
        if st.button("Trust Verdict", type="primary", use_container_width=True):
            log_event(product["id"], "trust_verdict", {"score": score, "label": label})
            st.success("Verdict trusted. The item is now purchase-ready.")
    with decision_mid:
        if st.button("Add to Bag", use_container_width=True):
            log_event(product["id"], "add_to_bag", {"source": "verdict_card", "score": score})
            st.success("Added to bag based on the verdict.")
    with decision_right:
        if st.button("Compare Again", use_container_width=True):
            log_event(product["id"], "compare_again", {"similar_items": product.get("similar_saved_items", [])})
            st.info("Comparison request captured. In production, this would open a side-by-side saved-item comparison.")

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
    metric_mid_left.metric("Trust clicks", int((df["event_name"] == "trust_verdict").sum()))
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
- Trust Verdict rate
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
    if st.session_state.get("pending_view"):
        st.session_state["active_view"] = st.session_state["pending_view"]
        st.session_state["nav_view"] = st.session_state["pending_view"]
        st.session_state["pending_view"] = None
    products = load_products()

    render_header(products)
    render_navigation()

    selected_id = st.session_state.get("selected_product_id")
    selected_product = next((product for product in products if product["id"] == selected_id), None)

    if st.session_state["active_view"] == VIEW_WISHLIST:
        render_wishlist(products)
    elif st.session_state["active_view"] == VIEW_VERDICT:
        if not selected_product:
            st.info("Open a verdict card from the wishlist to see the detailed decision view.")
            if st.button("Go to Wishlist"):
                st.session_state["active_view"] = VIEW_WISHLIST
                st.rerun()
        else:
            render_verdict(selected_product)
    elif st.session_state["active_view"] == VIEW_ANALYTICS:
        render_analytics()
    else:
        render_pm_notes()


if __name__ == "__main__":
    main()
