# Myntra Maturation Layer MVP

## Vercel deployment

This branch contains a Vercel-compatible FastAPI application. Deploy the repository as a Vercel Python project; Vercel detects the top-level `app` in `app.py` and serves the browser client from `public/index.html`.

For local use:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. The original Streamlit version is retained as `streamlit_app.py`.

A functional browser MVP for the Myntra wishlist case study.

The MVP simulates how Myntra's **Maturation Layer** can:

1. Diagnose the user's hesitation after they save a product.
2. Build a Case File from review, return, similar-buyer, and comparison signals.
3. Surface a Verdict Card inside the wishlist.
4. Capture Trust / Add-to-Bag / Override actions.
5. Track simple MVP analytics.

## Stack

- FastAPI and a static browser client
- Python
- SQLite event tracking
- JSON mock product data

## Legacy Streamlit prototype

The original Streamlit implementation is retained as `streamlit_app.py` for reference. The Vercel branch deploys the FastAPI version described above.

## Prototype scope

This is not a Myntra integration. It uses:

- Mock Myntra-like products
- Mock behavioral signals
- Sample reviews
- Sample return notes
- Sample similar-buyer evidence

## PM framing

Use this in the deck:

> A functional prototype that simulates how Myntra's Maturation Layer diagnoses hesitation, builds a Case File from shopper signals, and surfaces a Verdict Card inside the wishlist to help users move from “I like this” to “I am confident buying this.”

## Events tracked

- Verdict viewed
- Add to Cart clicked
- Buy Now clicked
- Comparison opened
