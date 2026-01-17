from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ---- imports from your package ----
# Requires: source/__init__.py exists
from source.rw import rw_forecast_table, rw_past_table  # noqa: E402


st.set_page_config(page_title="USD/ZAR Random Walk", layout="wide")
st.title("USD/ZAR Random Walk (Normal or Bootstrap)")
st.caption("Forecast and backtest tables from a simple RW model.")

# -----------------------------
# Sidebar: Data
# -----------------------------
st.sidebar.header("Data")

uploaded = st.sidebar.file_uploader("Upload CSV (date, usd_zar)", type=["csv"])
default_csv = Path("data") / "usd_zar_daily.csv"
use_default = st.sidebar.checkbox(
    "Use default CSV from ./data (if no upload)", value=True
)

@st.cache_data
def load_series_from_df(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    if "date" not in df.columns or "usd_zar" not in df.columns:
        raise ValueError("CSV must have columns: 'date' and 'usd_zar'")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["usd_zar"].astype(float).dropna()

series: pd.Series | None = None

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        series = load_series_from_df(df)
    elif use_default and default_csv.exists():
        df = pd.read_csv(default_csv)
        series = load_series_from_df(df)
    else:
        st.info("Upload a CSV or place a default file at ./data/usd_zar_daily.csv")
        st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

assert series is not None
series = series.dropna().sort_index()

last_date = series.index[-1]
S0 = float(series.iloc[-1])

st.sidebar.markdown(f"**Latest date in data:** {last_date.date().isoformat()}")
st.sidebar.markdown(f"**Latest S0:** {S0:.4f}")

# -----------------------------
# Sidebar: Global model controls
# -----------------------------
st.sidebar.header("Model")

method = st.sidebar.selectbox("Method", ["normal", "bootstrap"], index=0)
drift = st.sidebar.checkbox("Include drift (μ)", value=False)

st.sidebar.header("Quantiles")
quantile_choices = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
quantiles = st.sidebar.multiselect(
    "Quantiles",
    quantile_choices,
    default=[0.05, 0.25, 0.50, 0.75, 0.95],
)
quantiles = tuple(sorted(set(quantiles)))
if len(quantiles) == 0:
    st.sidebar.error("Pick at least one quantile.")
    st.stop()

n_sims = 20000
if method == "bootstrap":
    n_sims = st.sidebar.number_input("n_sims (bootstrap)", 1000, 500000, 20000, 1000)

st.sidebar.header("Forecast extra")
use_prob = st.sidebar.checkbox("Compute P(S > K)", value=False)
K = None
if use_prob:
    K = st.sidebar.number_input("K", value=float(S0), step=0.1)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Forecast", "Past (backtest)"])

with tab1:
    st.subheader("Forecast table")
    st.write("Forecast starts from the latest date/value in the loaded series.")

    # Window slider FOR FORECAST (max 1000, limited by available data)
    max_window_forecast = min(1000, max(0, len(series) - 2))
    if max_window_forecast < 60:
        st.error("Not enough data to support a 60-day window.")
        st.stop()

    window_forecast = st.slider(
        "Lookback window (past trading days)",
        min_value=60,
        max_value=max_window_forecast,
        value=min(252, max_window_forecast),
        step=10,
        key="window_forecast",
        help="Number of past daily returns used to calibrate the model (normal μ/σ or bootstrap sampling pool).",
    )

    default_targets = [
        (last_date + pd.offsets.BDay(5)).date().isoformat(),
        (last_date + pd.offsets.BDay(20)).date().isoformat(),
        (last_date + pd.offsets.BDay(60)).date().isoformat(),
    ]
    targets_text = st.text_area(
        "Target dates (one per line, YYYY-MM-DD)",
        value="\n".join(default_targets),
        height=110,
    )
    target_dates = [t.strip() for t in targets_text.splitlines() if t.strip()]

    colA, colB = st.columns([1, 1])
    run_forecast = colA.button("Run forecast", type="primary")
    if colB.button("Clear forecast output"):
        st.session_state.pop("forecast_tbl", None)

    if run_forecast:
        try:
            tbl = rw_forecast_table(
                series=series,
                target_dates=target_dates,
                window=int(window_forecast),
                drift=bool(drift),
                quantiles=quantiles,
                prob_gt=K if use_prob else None,
                method=method,
                n_sims=int(n_sims),
            )
            st.session_state["forecast_tbl"] = tbl
        except Exception as e:
            st.error(str(e))

    if "forecast_tbl" in st.session_state:
        tbl = st.session_state["forecast_tbl"]
        st.dataframe(tbl, use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            data=tbl.to_csv(index=False).encode("utf-8"),
            file_name="rw_forecast_table.csv",
            mime="text/csv",
        )

with tab2:
    st.subheader("Past table (backtest)")
    st.write("Pick a start date (S0 date) and past target dates to see model vs actual.")

    start_default = (last_date - pd.offsets.BDay(260)).date()
    start_date = st.date_input("Past start date", value=start_default)

    # Window slider FOR PAST (max 1000, but must be <= returns available up to start_date)
    series_upto = series.loc[:pd.Timestamp(start_date)]
    n_rets_upto = max(0, len(series_upto) - 1)  # returns = prices - 1

    max_window_past = min(1000, n_rets_upto)
    if max_window_past < 60:
        st.error(
            f"Not enough history before {pd.Timestamp(start_date).date().isoformat()} "
            f"to use a 60-day window. Available returns: {n_rets_upto}."
        )
        st.stop()

    window_past = st.slider(
        "Lookback window (past trading days)",
        min_value=60,
        max_value=max_window_past,
        value=min(252, max_window_past),
        step=10,
        key="window_past",
        help="Number of past daily returns used to calibrate the model (normal μ/σ or bootstrap sampling pool).",
)

    default_past_targets = [
        (pd.Timestamp(start_date) + pd.offsets.BDay(5)).date().isoformat(),
        (pd.Timestamp(start_date) + pd.offsets.BDay(20)).date().isoformat(),
        (pd.Timestamp(start_date) + pd.offsets.BDay(60)).date().isoformat(),
    ]
    past_targets_text = st.text_area(
        "Past target dates (one per line, YYYY-MM-DD)",
        value="\n".join(default_past_targets),
        height=110,
    )
    past_target_dates = [t.strip() for t in past_targets_text.splitlines() if t.strip()]

    colA, colB = st.columns([1, 1])
    run_past = colA.button("Run past table", type="primary")
    if colB.button("Clear past output"):
        st.session_state.pop("past_tbl", None)

    if run_past:
        try:
            tbl2 = rw_past_table(
                series=series,
                start_date=pd.Timestamp(start_date),
                target_dates=past_target_dates,
                window=int(window_past),
                drift=bool(drift),
                quantiles=quantiles,
                method=method,
                n_sims=int(n_sims),
            )
            st.session_state["past_tbl"] = tbl2
        except Exception as e:
            st.error(str(e))

    if "past_tbl" in st.session_state:
        tbl2 = st.session_state["past_tbl"]
        st.dataframe(tbl2, use_container_width=True)
        st.download_button(
            "Download past CSV",
            data=tbl2.to_csv(index=False).encode("utf-8"),
            file_name="rw_past_table.csv",
            mime="text/csv",
        )

st.divider()
st.caption("Next: add charts (history + fan chart) and a cleaner date-picker UI.")