from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt, log, exp
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def norm_cdf(x: float) -> float:
    """Standard normal CDF without scipy."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """
    Approx inverse CDF (ppf) using a rational approximation.
    Good enough for forecast bands (no scipy needed).
    Reference: Peter J. Acklam approximation (common implementation pattern).
    """
    # Coefficients
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    # Define break-points
    plow = 0.02425
    phigh = 1 - plow

    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


@dataclass(frozen=True)
class RWParams:
    mu: float      # mean daily log-return
    sigma: float   # std dev daily log-return
    window: int
    method: str    # "normal" or "bootstrap"
    drift: bool


def load_usdzar_csv(path: str | Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "usd_zar" not in df.columns:
        raise ValueError("Expected column 'usd_zar' in CSV.")
    return df["usd_zar"].astype(float)


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series).diff().dropna()


def estimate_rw_params(returns: pd.Series, window: int = 252, drift: bool = False) -> RWParams:
    if len(returns) < window:
        raise ValueError(f"Not enough data: have {len(returns)} returns, need >= {window}.")
    r = returns.iloc[-window:]
    mu = float(r.mean()) if drift else 0.0
    sigma = float(r.std(ddof=1))
    return RWParams(mu=mu, sigma=sigma, window=window, method="normal", drift=drift)


def horizon_steps(last_obs_date: pd.Timestamp, target_dates: Iterable[pd.Timestamp]) -> np.ndarray:
    """
    Convert target dates to 'steps' (approx business days) from last_obs_date.
    This uses pandas business day counting; it's a practical approximation.
    """
    last = pd.Timestamp(last_obs_date).normalize()
    hs = []
    for d in target_dates:
        d = pd.Timestamp(d).normalize()
        if d <= last:
            hs.append(0)
        else:
            # Count business days between (last, d], excluding last day itself
            bdays = pd.bdate_range(last + pd.Timedelta(days=1), d)
            hs.append(len(bdays))
    return np.array(hs, dtype=int)


def rw_log_distribution(S0: float, h: int, params: RWParams) -> tuple[float, float]:
    """
    Returns (mean_log, std_log) for log(S_{t+h}).
    """
    mean_log = log(S0) + h * params.mu
    std_log = sqrt(h) * params.sigma
    return mean_log, std_log


def rw_forecast_table(
    series: pd.Series,
    target_dates: Iterable[str | pd.Timestamp],
    window: int = 252,
    drift: bool = False,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    prob_gt: Optional[float] = None,
) -> pd.DataFrame:
    """
    Builds a table with forecast distribution summary for each target date.
    """
    series = series.dropna()
    S0 = float(series.iloc[-1])
    last_date = series.index[-1]

    rets = log_returns(series)
    params = estimate_rw_params(rets, window=window, drift=drift)

    tdates = [pd.Timestamp(d) for d in target_dates]
    hs = horizon_steps(last_date, tdates)

    rows = []
    for d, h in zip(tdates, hs):
        mean_log, std_log = rw_log_distribution(S0, int(h), params)
        qvals = {}
        for q in quantiles:
            z = norm_ppf(q)
            qvals[f"q{int(q*100):02d}"] = exp(mean_log + z * std_log) if h > 0 else S0

        p_gt = None
        if prob_gt is not None:
            if h == 0 or std_log == 0:
                p_gt = float(S0 > prob_gt)
            else:
                z = (log(prob_gt) - mean_log) / std_log
                p_gt = 1.0 - norm_cdf(z)

        rows.append(
            {
                "target_date": d.date().isoformat(),
                "h_steps": int(h),
                "S0": S0,
                "mu": params.mu,
                "sigma": params.sigma,
                **qvals,
                **({"P(S>K)": p_gt, "K": prob_gt} if prob_gt is not None else {}),
            }
        )

    return pd.DataFrame(rows)