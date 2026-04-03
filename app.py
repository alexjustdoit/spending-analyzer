import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from utils.parser import parse_csv, prepare_transactions
from utils.analyzer import categorize_transactions, generate_summary, CATEGORY_COLORS, CATEGORIES

load_dotenv()

st.set_page_config(page_title="Spending Analyzer", page_icon="💸", layout="wide")

# ── Client ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_openai() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ai = get_openai()

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_currency(val: float) -> str:
    return f"${val:,.2f}"


def get_date_range_str(df: pd.DataFrame) -> str:
    if df["date"].isna().all():
        return "Unknown date range"
    min_d = df["date"].min()
    max_d = df["date"].max()
    if min_d.month == max_d.month and min_d.year == max_d.year:
        return min_d.strftime("%B %Y")
    return f"{min_d.strftime('%b %d')} – {max_d.strftime('%b %d, %Y')}"


def color_for_category(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, "#94a3b8")

# ── Charts ────────────────────────────────────────────────────────────────────

def chart_donut(df: pd.DataFrame) -> go.Figure:
    totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    colors = [color_for_category(c) for c in totals.index]
    fig = go.Figure(go.Pie(
        labels=totals.index,
        values=totals.values,
        hole=0.55,
        marker_colors=colors,
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    return fig


def chart_category_bar(df: pd.DataFrame) -> go.Figure:
    totals = df.groupby("category")["amount"].sum().sort_values(ascending=True)
    colors = [color_for_category(c) for c in totals.index]
    fig = go.Figure(go.Bar(
        x=totals.values,
        y=totals.index,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>$%{x:,.2f}<extra></extra>",
        text=[fmt_currency(v) for v in totals.values],
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=0, r=60),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(250, len(totals) * 40),
    )
    return fig


def chart_over_time(df: pd.DataFrame) -> go.Figure | None:
    if df["date"].isna().all():
        return None
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = df.groupby("week")["amount"].sum().reset_index()
    fig = px.bar(
        weekly,
        x="week",
        y="amount",
        labels={"week": "", "amount": "Amount ($)"},
        color_discrete_sequence=["#3b82f6"],
    )
    fig.update_traces(hovertemplate="Week of %{x|%b %d}<br>$%{y:,.2f}<extra></extra>")
    fig.update_layout(
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        height=250,
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple:
    """Returns (df_clean, date_range_str) or (None, None) if not ready."""
    with st.sidebar:
        st.title("💸 Spending Analyzer")
        st.caption("Upload a CSV export from your bank or credit card.")
        st.divider()

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

        if not uploaded:
            st.markdown(
                "**Supported formats:**\n"
                "- Chase, Bank of America, Capital One\n"
                "- Any CSV with date, description, and amount columns\n\n"
                "Export from your bank's transaction history page."
            )
            return None, None

        try:
            raw_df, mapping = parse_csv(uploaded)
        except ValueError as e:
            st.error(str(e))
            return None, None

        st.divider()
        st.markdown("**Column mapping**")

        all_cols = raw_df.columns.tolist()

        date_col = st.selectbox(
            "Date column",
            options=["(none)"] + all_cols,
            index=(all_cols.index(mapping["date"]) + 1) if mapping["date"] else 0,
        )
        desc_col = st.selectbox(
            "Description column",
            options=all_cols,
            index=all_cols.index(mapping["description"]) if mapping["description"] else 0,
        )
        amount_col = st.selectbox(
            "Amount column",
            options=all_cols,
            index=all_cols.index(mapping["amount"]) if mapping["amount"] else 0,
        )

        final_mapping = {
            "date": date_col if date_col != "(none)" else None,
            "description": desc_col,
            "amount": amount_col,
        }

        st.divider()
        exclude_credits = st.toggle("Exclude credits & refunds", value=True)

        try:
            df_clean = prepare_transactions(raw_df, final_mapping, exclude_credits=exclude_credits)
        except Exception as e:
            st.error(f"Could not parse transactions: {e}")
            return None, None

        if df_clean.empty:
            st.error("No transactions found after parsing. Try adjusting the column mapping or credit filter.")
            return None, None

        # Date range filter
        if not df_clean["date"].isna().all():
            st.divider()
            st.markdown("**Date filter**")
            min_date = df_clean["date"].min().date()
            max_date = df_clean["date"].max().date()
            start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)
            df_clean = df_clean[
                (df_clean["date"].dt.date >= start_date) &
                (df_clean["date"].dt.date <= end_date)
            ]

        st.divider()
        st.caption(f"{len(df_clean)} transactions loaded")

        if st.button("Analyze", type="primary", use_container_width=True):
            st.session_state.analyze = True
            st.session_state.df_input = df_clean

        return df_clean, get_date_range_str(df_clean)

# ── Main area ─────────────────────────────────────────────────────────────────

def render_welcome():
    st.markdown(
        """
        <div style="text-align:center;padding:4rem 0 2rem 0;color:#6b7280;">
            <div style="font-size:5rem">💸</div>
            <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;margin-top:1rem">
                Spending Analyzer
            </div>
            <div style="margin-top:0.5rem;font-size:1.05rem">
                Upload a bank or credit card CSV export to see where your money is going.
            </div>
            <div style="margin-top:2rem;font-size:0.9rem;max-width:480px;margin-left:auto;margin-right:auto;line-height:1.8">
                AI categorizes every transaction · Charts by category and over time · Plain-English summary
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(df: pd.DataFrame, date_range: str):
    # ── Run analysis ──
    with st.spinner("Categorizing transactions…"):
        df = categorize_transactions(df, ai)

    with st.spinner("Generating summary…"):
        summary = generate_summary(df, date_range, ai)

    # ── Metrics ──
    total = df["amount"].sum()
    top_cat = df.groupby("category")["amount"].sum().idxmax()
    top_cat_amt = df.groupby("category")["amount"].sum().max()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Spend", fmt_currency(total))
    m2.metric("Transactions", len(df))
    m3.metric("Top Category", top_cat)
    m4.metric("Date Range", date_range)

    st.divider()

    # ── Donut + Summary ──
    col_chart, col_summary = st.columns([1, 1])
    with col_chart:
        st.subheader("By Category")
        st.plotly_chart(chart_donut(df), use_container_width=True)
    with col_summary:
        st.subheader("AI Summary")
        st.markdown(
            f'<div style="background:#1e293b;border-radius:12px;padding:1.25rem 1.5rem;'
            f'line-height:1.7;color:#e2e8f0;margin-top:0.5rem">{summary}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Category breakdown bar ──
    st.subheader("Breakdown")
    st.plotly_chart(chart_category_bar(df), use_container_width=True)

    st.divider()

    # ── Over time ──
    time_fig = chart_over_time(df)
    if time_fig:
        st.subheader("Spending Over Time")
        st.plotly_chart(time_fig, use_container_width=True)
        st.divider()

    # ── Top merchants ──
    col_merch, col_cats = st.columns([1, 1])
    with col_merch:
        st.subheader("Top Merchants")
        top_merchants = (
            df.groupby("description")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_merchants.columns = ["Merchant", "Total"]
        top_merchants["Total"] = top_merchants["Total"].apply(fmt_currency)
        st.dataframe(top_merchants, hide_index=True, use_container_width=True)

    with col_cats:
        st.subheader("Category Totals")
        cat_totals = (
            df.groupby("category")["amount"]
            .agg(["sum", "count"])
            .sort_values("sum", ascending=False)
            .reset_index()
        )
        cat_totals.columns = ["Category", "Total", "# Transactions"]
        cat_totals["% of Spend"] = (cat_totals["Total"] / total * 100).round(1).astype(str) + "%"
        cat_totals["Total"] = cat_totals["Total"].apply(fmt_currency)
        st.dataframe(cat_totals, hide_index=True, use_container_width=True)

    st.divider()

    # ── Full transaction table ──
    with st.expander(f"All transactions ({len(df)})"):
        display_df = df.copy()
        if not display_df["date"].isna().all():
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df["amount"] = display_df["amount"].apply(fmt_currency)
        display_df = display_df.rename(columns={
            "date": "Date", "description": "Description",
            "amount": "Amount", "category": "Category"
        })
        cols = ["Date", "Description", "Amount", "Category"] if "Date" in display_df.columns else ["Description", "Amount", "Category"]
        st.dataframe(display_df[cols], hide_index=True, use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "analyze" not in st.session_state:
        st.session_state.analyze = False

    df_clean, date_range = render_sidebar()

    if not st.session_state.analyze or df_clean is None:
        render_welcome()
        return

    render_results(st.session_state.df_input, date_range)


if __name__ == "__main__":
    main()
