from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests


FRANKFURTER_BASE = "https://api.frankfurter.app"  # public instance :contentReference[oaicite:3]{index=3}


@dataclass(frozen=True)
class FXSeries:
    base: str = "USD"
    quote: str = "ZAR"


def _daterange_year_chunks(start: date, end: date) -> Iterable[tuple[date, date]]:
    """Yield inclusive date chunks no longer than ~1 year to keep responses small."""
    cur = start
    while cur <= end:
        nxt = min(date(cur.year, 12, 31), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)

def _call_timeseries(start: date, end: date, series: FXSeries) -> dict:
    """
    Frankfurter time series endpoint uses /YYYY-MM-DD..YYYY-MM-DD :contentReference[oaicite:4]{index=4}
    We prefer from/to params because latest?from=USD works (base USD) :contentReference[oaicite:5]{index=5}.
    If that fails, we fall back to base/symbols style (used by newer docs/instances). :contentReference[oaicite:6]{index=6}
    """
    url = f"{FRANKFURTER_BASE}/{start.isoformat()}..{end.isoformat()}"
    # Try legacy params first
    params_primary = {"from": series.base, "to": series.quote}
    r = requests.get(url, params=params_primary, timeout=30)
    if r.ok:
        return r.json()

    # Fallback style
    params_fallback = {"base": series.base, "symbols": series.quote}
    r2 = requests.get(url, params=params_fallback, timeout=30)
    r2.raise_for_status()
    return r2.json()

def _call_latest(series: FXSeries) -> dict:
    """
    Fetch latest available (working-day) rate from Frankfurter.
    Tries legacy from/to first, then base/symbols fallback.
    """
    url = f"{FRANKFURTER_BASE}/latest"

    params_primary = {"from": series.base, "to": series.quote}
    r = requests.get(url, params=params_primary, timeout=30)
    if r.ok:
        return r.json()

    params_fallback = {"base": series.base, "symbols": series.quote}
    r2 = requests.get(url, params=params_fallback, timeout=30)
    r2.raise_for_status()
    return r2.json()


def latest_available_date(series: FXSeries) -> date:
    payload = _call_latest(series)
    # Frankfurter returns a "date" field like "YYYY-MM-DD"
    return pd.to_datetime(payload["date"]).date()

def fetch_usdzar_daily(start: date, end: Optional[date] = None) -> pd.DataFrame:
    end = end or latest_available_date(series)
    series = FXSeries("USD", "ZAR")

    frames = []
    for s, e in _daterange_year_chunks(start, end):
        payload = _call_timeseries(s, e, series)

        rates = payload.get("rates", {})
        # rates is { "YYYY-MM-DD": { "ZAR": <rate> }, ... }
        rows = []
        for d_str, quote_map in rates.items():
            if series.quote in quote_map:
                rows.append((pd.to_datetime(d_str), float(quote_map[series.quote])))

        df = pd.DataFrame(rows, columns=["date", "usd_zar"]).sort_values("date")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    out = out.set_index("date")
    return out


def save_series_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def load_series_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def update_local_series(path: Path, start_if_missing: date) -> pd.DataFrame:
    series = FXSeries("USD", "ZAR")
    latest = latest_available_date(series)  # latest working-day in Frankfurter

    if path.exists():
        df = load_series_csv(path)
        last_date = df.index.max().date()

        new_start = last_date + timedelta(days=1)
        if new_start <= latest:
            df_new = fetch_usdzar_daily(new_start, latest)
            df = pd.concat([df, df_new]).sort_index().drop_duplicates()
            save_series_csv(df, path)

        return df

    df = fetch_usdzar_daily(start_if_missing, latest)
    save_series_csv(df, path)
    return df


if __name__ == "__main__":
    data_path = Path("data/usd_zar_daily.csv")
    df = update_local_series(data_path, start_if_missing=date(2016, 1, 1))
    print(df.tail())