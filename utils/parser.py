import re
import pandas as pd

DATE_KEYWORDS = ["date", "transaction date", "posted", "posting date", "trans date", "time"]
DESC_KEYWORDS = ["description", "merchant", "payee", "name", "memo", "details", "narrative", "transaction"]
AMOUNT_KEYWORDS = ["amount", "debit", "charge", "price", "sum", "total", "transaction amount"]

PAYMENT_PATTERNS = re.compile(
    r"payment|thank you|autopay|credit card payment|online payment|mobile payment",
    re.IGNORECASE,
)


def _detect_column(columns: list[str], keywords: list[str]) -> str | None:
    cols_lower = {c.lower().strip(): c for c in columns}
    for kw in keywords:
        for col_lower, col_orig in cols_lower.items():
            if kw in col_lower:
                return col_orig
    return None


def parse_csv(file) -> tuple[pd.DataFrame, dict]:
    """
    Parse uploaded CSV file.
    Returns (raw_df, column_mapping) where mapping keys are date/description/amount.
    Values are column names if detected, else None.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV appears empty or has too few columns.")

    mapping = {
        "date": _detect_column(df.columns.tolist(), DATE_KEYWORDS),
        "description": _detect_column(df.columns.tolist(), DESC_KEYWORDS),
        "amount": _detect_column(df.columns.tolist(), AMOUNT_KEYWORDS),
    }
    return df, mapping


def prepare_transactions(df: pd.DataFrame, mapping: dict, exclude_credits: bool = True) -> pd.DataFrame:
    """
    Build a clean transactions dataframe.
    Returns df with standardized columns: date, description, amount
    """
    result = pd.DataFrame()

    # Amount
    amount_col = mapping["amount"]
    raw_amounts = (
        df[amount_col]
        .astype(str)
        .str.replace(r"[$£€,\s]", "", regex=True)
        .str.replace(r"\((.+?)\)", r"-\1", regex=True)  # (123.45) → -123.45
    )
    result["amount"] = pd.to_numeric(raw_amounts, errors="coerce")

    # Description
    result["description"] = df[mapping["description"]].astype(str).str.strip()

    # Date
    if mapping.get("date"):
        result["date"] = pd.to_datetime(df[mapping["date"]], infer_datetime_format=True, errors="coerce")
    else:
        result["date"] = pd.NaT

    result = result.dropna(subset=["amount", "description"])
    result = result[result["description"].str.len() > 0]

    # Determine debit convention: if majority of amounts are positive, positive = charge
    # if majority are negative, negative = charge
    if exclude_credits:
        pos_count = (result["amount"] > 0).sum()
        neg_count = (result["amount"] < 0).sum()
        if pos_count >= neg_count:
            result = result[result["amount"] > 0]
        else:
            result = result[result["amount"] < 0]
            result["amount"] = result["amount"].abs()

        # Also filter out card payments to self
        result = result[~result["description"].str.match(PAYMENT_PATTERNS)]

    result["amount"] = result["amount"].abs()
    result = result.reset_index(drop=True)
    return result
