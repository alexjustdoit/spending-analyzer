# 💸 Spending Analyzer

Upload a bank or credit card CSV export — AI categorizes every transaction and shows you where your money is going.

**Live app:** https://spending-analyzer.streamlit.app

---

## Features

- **Auto column detection** — recognizes date, description, and amount columns from most bank CSV formats
- **AI categorization** — batches unique merchants to gpt-5.4-nano for efficient, accurate categorization
- **Charts** — donut by category, horizontal bar breakdown, weekly spending over time
- **Top merchants** — ranked by total spend
- **AI summary** — plain-English 3-4 sentence analysis of your spending patterns
- **Credit filter** — automatically excludes payments and refunds
- **Date range filter** — analyze a specific period when multiple months are present
- **Full transaction table** — every transaction with its assigned category

## Supported CSV formats

- Chase, Bank of America, Capital One
- Any CSV with date, description, and amount columns
- Manual column mapping available if auto-detection misses

## Stack

- **Frontend:** Streamlit (deployed on Streamlit Community Cloud)
- **AI:** OpenAI `gpt-5.4-nano`
- **Charts:** Plotly
- **Data:** pandas
- **Keepalive:** GitHub Actions (pings every 5 hours)

## Setup

### 1. Environment variables

Copy `.env.example` to `.env`:

```
OPENAI_API_KEY=your-openai-key
```

### 2. Install and run locally

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy to Streamlit Community Cloud

- Connect your GitHub repo at [share.streamlit.io](https://share.streamlit.io)
- Set `OPENAI_API_KEY` as a secret
- Add your app URL as `APP_URL` in GitHub Actions secrets for the keepalive

## Backlog

- Subscription detection — identify recurring charges and calculate annual burn
- Month-over-month comparison
- Export categorized CSV
- Custom category rules
