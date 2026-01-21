from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt

from source.rw import rw_forecast_table, rw_past_table  # noqa: E402
from source.frankfurter_fetch import update_local_series


st.set_page_config(page_title="USD/ZAR Random Walk", layout="wide")
st.title("USD/ZAR Random Walk (Normal or Bootstrap)")
st.caption("Forecast and backtest tables from a simple RW model.")

st.sidebar.header("Data")

uploaded = st.sidebar.file_uploader("Upload CSV (date, usd_zar)", type=["csv"])
default_csv = Path("data") / "usd_zar_daily.csv"
use_default = st.sidebar.checkbox("Use default CSV from ./data (if no upload)", value=True)


@st.cache_data
def load_series_from_df(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    if "date" not in df.columns or "usd_zar" not in df.columns:
        raise ValueError("CSV must have columns: 'date' and 'usd_zar'")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["usd_zar"].astype(float).dropna()


def df_to_series(df: pd.DataFrame) -> pd.Series:
    s = load_series_from_df(df)
    return s.dropna().sort_index()


if "series" not in st.session_state:
    series: pd.Series | None = None
    try:
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            series = df_to_series(df)
            st.session_state["data_source"] = "upload"
        elif use_default and default_csv.exists():
            df = pd.read_csv(default_csv)
            series = df_to_series(df)
            st.session_state["data_source"] = "default_csv"
        else:
            st.info("Upload a CSV or place a default file at ./data/usd_zar_daily.csv")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    st.session_state["series"] = series

series = st.session_state["series"]
series = series.dropna().sort_index()

last_date = series.index[-1]
S0 = float(series.iloc[-1])

st.sidebar.markdown(f"**Latest date in data:** {last_date.date().isoformat()}")
st.sidebar.markdown(f"**Latest S0:** {S0:.4f}")

st.sidebar.subheader("Update data")

if st.sidebar.button("Update data from Frankfurter"):
    try:
        df_updated = update_local_series(default_csv, start_if_missing=date(2016, 1, 1))
        st.session_state["series"] = df_to_series(df_updated.reset_index())
        series = st.session_state["series"]
        st.sidebar.success(f"Updated to {series.index[-1].date().isoformat()}")
    except Exception as e:
        st.sidebar.error(f"Update failed: {e}")

df_download = series.rename("usd_zar").reset_index().rename(columns={"index": "date"})
st.sidebar.download_button(
    "Download updated CSV",
    data=df_download.to_csv(index=False).encode("utf-8"),
    file_name="usd_zar_daily_updated.csv",
    mime="text/csv",
)

st.subheader("USD/ZAR history")
hist_years = st.selectbox("History range", ["6M", "1Y", "5Y", "All"], index=1)

end = series.index[-1]
if hist_years == "6M":
    start = end - pd.DateOffset(months=6)
elif hist_years == "1Y":
    start = end - pd.DateOffset(years=1)
elif hist_years == "5Y":
    start = end - pd.DateOffset(years=5)
else:
    start = series.index[0]

st.line_chart(series.loc[start:end])

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

tab1, tab2 = st.tabs(["Forecast", "Past (backtest)"])

with tab1:
    st.subheader("Forecast table")
    st.write("Forecast starts from the latest date/value in the loaded series.")

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
        ft = tbl.copy()
        ft["target_date"] = pd.to_datetime(ft["target_date"])
        ft = ft.sort_values("target_date")

        needed = {"q05", "q25", "q50", "q75", "q95"}
        if needed.issubset(set(ft.columns)):
            st.subheader("Forecast fan chart")

            fig, ax = plt.subplots()
            ax.plot(ft["target_date"], ft["q50"], label="Median")
            ax.fill_between(ft["target_date"], ft["q05"], ft["q95"], alpha=0.2, label="5–95%")
            ax.fill_between(ft["target_date"], ft["q25"], ft["q75"], alpha=0.3, label="25–75%")
            ax.set_xlabel("Date")
            ax.set_ylabel("USD/ZAR")
            ax.legend()

            st.pyplot(fig)
        else:
            st.info("Fan chart needs quantiles q05, q25, q50, q75, q95. Select these quantiles to display the chart.")

with tab2:
    st.subheader("Past table (backtest)")
    st.write("Pick a start date (S0 date) and past target dates to see model vs actual.")

    start_default = (last_date - pd.offsets.BDay(260)).date()
    start_date = st.date_input("Past start date", value=start_default)

    series_upto = series.loc[:pd.Timestamp(start_date)]
    n_rets_upto = max(0, len(series_upto) - 1)

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

with st.expander("About the random walk models"):
    st.markdown(r"""
**What this app does**

This app builds **forecast** and **backtest** tables for USD/ZAR using a **random-walk (RW) model**
on **daily log-returns**.

**Returns & window**
- Daily log-return:  $r_t = \ln(S_t) - \ln(S_{t-1}) = \ln(S_t / S_{t-1})$
- **Lookback window** = number of past trading days of returns used to calibrate the model
  (estimate parameters or define the bootstrap sampling pool).

**Normal model**
Assumes log-returns are Normal:
- $r_t \sim N(\mu,\sigma^2)$
- $\mu$ is the mean return over the lookback window if **drift** is enabled, otherwise $\mu = 0$
- $\sigma$ is the standard deviation of returns over the lookback window  
Then:
- $\ln S_{h} = \ln(S_0) + \sum_{i=1}^{h} r_{i} \sim N(\ln S_0 + \mu h,\; \sigma^2 h)$
So:
- Quantile: $Q_p(S_h)=\exp\!\big(\ln S_0 + \mu h + \sigma\sqrt{h}\,\Phi^{-1}(p)\big)$
- Tail probability: $P(S_h>K)=1-\Phi\!\left(\dfrac{\ln K-(\ln S_0+\mu h)}{\sigma\sqrt{h}}\right)$
where $\Phi(\cdot)$ is the **standard normal CDF** and $\Phi^{-1}(\cdot)$ is its inverse (the z-score / ppf).

**Bootstrap model**
Does not assume Normality. Instead, it resamples past returns (with replacement):
- We form **shocks**: $\epsilon_t = r_t - \bar r$
- If **drift** is enabled we add back $\bar r \cdot h$; if not, drift is 0
- Quantiles/probabilities are estimated from many simulated outcomes.

**Outputs**
The tables report forecast quantiles (and optional probability $P(S>K)$) and, for the past table,
the **actual** observed value on the target date.
""")

