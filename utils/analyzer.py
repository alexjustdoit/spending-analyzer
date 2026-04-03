import json
import pandas as pd
from openai import OpenAI

MODEL = "gpt-5.4-nano"

CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transport",
    "Shopping",
    "Entertainment & Streaming",
    "Health & Fitness",
    "Travel",
    "Utilities & Bills",
    "Personal Care",
    "Other",
]

CATEGORY_COLORS = {
    "Food & Dining": "#f97316",
    "Groceries": "#22c55e",
    "Transport": "#3b82f6",
    "Shopping": "#a855f7",
    "Entertainment & Streaming": "#ec4899",
    "Health & Fitness": "#14b8a6",
    "Travel": "#f59e0b",
    "Utilities & Bills": "#6b7280",
    "Personal Care": "#84cc16",
    "Other": "#94a3b8",
}


def categorize_transactions(df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """
    Categorize transactions by merchant name.
    Deduplicates merchants before sending to LLM to minimize tokens.
    Returns df with added 'category' column.
    """
    unique_merchants = df["description"].unique().tolist()

    # Batch in chunks of 150 to stay within context limits
    category_map = {}
    chunk_size = 150
    for i in range(0, len(unique_merchants), chunk_size):
        chunk = unique_merchants[i : i + chunk_size]
        chunk_map = _categorize_batch(chunk, client)
        category_map.update(chunk_map)

    df = df.copy()
    df["category"] = df["description"].map(category_map).fillna("Other")
    return df


def _categorize_batch(merchants: list[str], client: OpenAI) -> dict[str, str]:
    merchants_text = "\n".join(f"{i+1}. {m}" for i, m in enumerate(merchants))
    categories_list = "\n".join(f"- {c}" for c in CATEGORIES)

    system = f"""Categorize each merchant/transaction description into one of these categories:
{categories_list}

Return a JSON object mapping each merchant name EXACTLY as provided to its category.
Return ONLY valid JSON, no markdown fences."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": merchants_text},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model adds them anyway
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        return json.loads(raw)
    except Exception:
        # Fallback: assign Other to all if LLM fails
        return {m: "Other" for m in merchants}


def generate_summary(df: pd.DataFrame, date_range: str, client: OpenAI) -> str:
    """Generate a plain-English spending analysis."""
    total = df["amount"].sum()
    category_totals = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )
    top_merchants = (
        df.groupby("description")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    breakdown = "\n".join(
        f"- {cat}: ${amt:,.2f} ({amt / total * 100:.0f}%)"
        for cat, amt in category_totals.items()
    )
    merchants_text = "\n".join(
        f"- {name}: ${amt:,.2f}"
        for name, amt in top_merchants.items()
    )

    prompt = f"""Here is a spending summary for {date_range}:

Total spend: ${total:,.2f} across {len(df)} transactions

By category:
{breakdown}

Top merchants:
{merchants_text}

Write a 3-4 sentence plain-English analysis. Be specific with dollar amounts and percentages. \
Note the biggest spending areas, any patterns worth reviewing, and one actionable observation. \
Keep the tone friendly and helpful, not judgmental."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Could not generate summary. Check your OpenAI API key."
