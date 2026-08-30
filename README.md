# Myntra Maturation Layer MVP

A functional Streamlit prototype for the Myntra wishlist case study.

The MVP simulates how Myntra's **Maturation Layer** can:

1. Diagnose the user's hesitation after they save a product.
2. Build a Case File from review, return, similar-buyer, and comparison signals.
3. Surface a Verdict Card inside the wishlist.
4. Capture Trust / Add-to-Bag / Override actions.
5. Track simple MVP analytics.

## Stack

- Streamlit
- Python
- Groq API
- SQLite
- JSON mock product data

## Setup

```bash
unzip myntra_maturation_mvp_groq.zip
cd myntra_maturation_mvp
pip install -r requirements.txt
cp .env.example .env
```

Add your Groq key in `.env`:

```bash
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Run:

```bash
streamlit run app.py
```

## Groq behavior

The app uses Groq to generate the Verdict Card Case File when `GROQ_API_KEY` is present.

Default model:

```text
llama-3.3-70b-versatile
```

You can change it in `.env`:

```bash
GROQ_MODEL=llama-3.1-8b-instant
```

If no Groq key is present, the app still works using deterministic fallback summaries.

## Prototype scope

This is not a Myntra integration. It uses:

- Mock Myntra-like products
- Mock behavioral signals
- Sample reviews
- Sample return notes
- Sample similar-buyer evidence

## PM framing

Use this in the deck:

> A functional prototype that simulates how Myntra's Maturation Layer diagnoses hesitation, builds a Case File using Groq, and surfaces a Verdict Card inside the wishlist to help users move from “I like this” to “I am confident buying this.”

## Events tracked

- Verdict viewed
- Trust Verdict clicked
- Add to Bag clicked
- Compare Again clicked
- Override Verdict clicked
- Simulate Maturation clicked
