from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt, log, exp
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def norm_cdf(x: float) -> float:
    """
    Standard normal CDF (Phi). Can handle any normal X ~ N(mu, sigma**2) by standardizing.
    Z = (X - mu)/sigma ~ N(0,1). F(X) = P(X<= x) = Phi((x-mu)/sigma)
    Qp = mu + sigma * Phi_inverse(p)
    We will have log(S_{t+h}) ~ N(log(S_t + h*mu, h*sigma**2)) where we only need our present position S_t
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0))) #more generally 0.5 * (1.0 + erf((x - mu) / sigma*sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """
    Approx inverse CDF (ppf) using a rational approximation. Note normal cdf has no closed form inverse.
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
    Returns (mean_log, std_log) for log(S_{t+h}). Note log(S_{t+h}) ~ N(log(S_t + h*mu, h*sigma**2))
    """
    mean_log = log(S0) + h * params.mu
    std_log = sqrt(h) * params.sigma
    return mean_log, std_log

def rw_bootstrap_quantiles(S0: float, r_window: pd.Series, h: int, n_sims: int = 20000,
                           quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
    # sample h returns for each simulation (with replacement)
    sims = np.random.choice(r_window.values, size=(n_sims, h), replace=True)
    log_ST = np.log(S0) + sims.sum(axis=1)
    ST = np.exp(log_ST)
    return {f"q{int(q*100):02d}": float(np.quantile(ST, q)) for q in quantiles}

def rw_forecast_table(
    series: pd.Series,
    target_dates: Iterable[str | pd.Timestamp],
    window: int = 252,
    drift: bool = False,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    prob_gt: Optional[float] = None,
    method: str = "normal",          # "normal" or "bootstrap"
    n_sims: int = 20000,             # only used for bootstrap
) -> pd.DataFrame:
    """
    Builds a table with forecast distribution summary for each target date.
    method:
      - "normal": log-returns ~ N(mu, sigma^2)
      - "bootstrap": sample historical log-returns with replacement
    """
    series = series.dropna()
    S0 = float(series.iloc[-1])
    last_date = series.index[-1]

    rets = log_returns(series)

    if len(rets) < window:
        raise ValueError(f"Not enough data: have {len(rets)} returns, need >= {window}.")

    # last `window` returns define the historical distribution we bootstrap from
    r_window = rets.iloc[-window:]

    window_mean = float(r_window.mean())
    window_std = float(r_window.std(ddof=1))

    # you can still report mu/sigma for reference (and use mu if drift=True)
    params = estimate_rw_params(rets, window=window, drift=drift)

    tdates = [pd.Timestamp(d) for d in target_dates]
    hs = horizon_steps(last_date, tdates)

    rows = []
    for d, h in zip(tdates, hs):
        h = int(h)

        qvals = {}
        p_gt = None

        if h == 0:
            # No time passes: distribution collapses at S0
            for q in quantiles:
                qvals[f"q{int(q*100):02d}"] = S0
            if prob_gt is not None:
                p_gt = float(S0 > prob_gt)

        else:
            if method == "bootstrap":
                # --- bootstrap distribution of S_T ---
                # If drift=True, add mu*h as a constant drift to the summed bootstrapped returns.
                # If drift=False, mu=0 and this does nothing.
                mu_hat = window_mean
                shocks = (r_window - mu_hat).values
                mu_used = mu_hat if drift else 0.0

                sims = np.random.choice(shocks, size=(n_sims, h), replace=True)
                log_ST = np.log(S0) + sims.sum(axis=1) + (mu_used * h)
                ST = np.exp(log_ST)

                for q in quantiles:
                    qvals[f"q{int(q*100):02d}"] = float(np.quantile(ST, q))

                if prob_gt is not None:
                    p_gt = float(np.mean(ST > prob_gt))

            elif method == "normal":
                # --- normal approximation for log(S_T) ---
                mean_log, std_log = rw_log_distribution(S0, h, params)

                for q in quantiles:
                    z = norm_ppf(q)
                    qvals[f"q{int(q*100):02d}"] = exp(mean_log + z * std_log)

                if prob_gt is not None:
                    if std_log == 0:
                        p_gt = float(S0 > prob_gt)
                    else:
                        z = (log(prob_gt) - mean_log) / std_log
                        p_gt = 1.0 - norm_cdf(z)

            else:
                raise ValueError("method must be 'normal' or 'bootstrap'")
        
        rows.append(
            {
                "current_date": pd.Timestamp(last_date).date().isoformat(),
                "target_date": d.date().isoformat(),
                "h_steps": h,
                "S0": S0,
                "window_mean": window_mean,
                "window_std": window_std,
                "method": method,
                **qvals,
                **({"P(S>K)": p_gt, "K": prob_gt} if prob_gt is not None else {}),
            }
        )

    return pd.DataFrame(rows)

def rw_past_table(
    series: pd.Series,
    start_date: str | pd.Timestamp,
    target_dates: Iterable[str | pd.Timestamp],
    window: int = 252,
    drift: bool = False,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    method: str = "normal",          # "normal" or "bootstrap"
    n_sims: int = 20000,             # only used for bootstrap
) -> pd.DataFrame:
    """
    Backtest-style table:
    - Treat `start_date` as the "current" date (S0 observed from data at/ before start_date).
    - For each past target date, compute h_steps from the data index (no business-day approximation).
    - Compute forecast distribution using the chosen method, calibrated using returns *up to start_date*.
    - Add 'actual' = observed series value at/ before the target date.
    Idea is to be able to showcase our RW-based models on past data.
    """
    series = series.dropna().sort_index()

    # --- helper: get last observed (date, value) on or before a timestamp ---
    def last_obs_on_or_before(ts: pd.Timestamp) -> tuple[pd.Timestamp, float]:
        ts = pd.Timestamp(ts)
        s = series.loc[:ts]
        if s.empty:
            raise ValueError(f"No observations on or before {ts.date().isoformat()}.")
        obs_date = s.index[-1]
        obs_val = float(s.iloc[-1])
        return obs_date, obs_val

    start_ts = pd.Timestamp(start_date)
    start_obs_date, S0 = last_obs_on_or_before(start_ts)

    # returns up to the start observation date (so the model only uses info available then)
    rets_full = log_returns(series.loc[:start_obs_date])

    if len(rets_full) < window:
        raise ValueError(
            f"Not enough data before start_date={start_obs_date.date().isoformat()}: "
            f"have {len(rets_full)} returns, need >= {window}."
        )

    r_window = rets_full.iloc[-window:]
    window_mean = float(r_window.mean())
    window_std = float(r_window.std(ddof=1))

    params = estimate_rw_params(rets_full, window=window, drift=drift)

    # Precompute positions for fast h_steps based on index locations
    # (h_steps = number of return steps between start_obs_date and target_obs_date)
    idx = series.index
    start_pos = idx.get_loc(start_obs_date)

    rows = []
    for td in target_dates:
        target_ts = pd.Timestamp(td)
        target_obs_date, actual = last_obs_on_or_before(target_ts)

        # compute steps directly from index positions (no business day approx)
        target_pos = idx.get_loc(target_obs_date)
        h = int(max(0, target_pos - start_pos))  # if target before start, clamp to 0

        qvals = {}

        if h == 0:
            for q in quantiles:
                qvals[f"q{int(q*100):02d}"] = S0
        else:
            if method == "bootstrap":
                mu_hat = window_mean
                shocks = (r_window - mu_hat).values
                mu_used = mu_hat if drift else 0.0

                sims = np.random.choice(shocks, size=(n_sims, h), replace=True)
                log_ST = np.log(S0) + sims.sum(axis=1) + (mu_used * h)
                ST = np.exp(log_ST)

                for q in quantiles:
                    qvals[f"q{int(q*100):02d}"] = float(np.quantile(ST, q))

            elif method == "normal":
                mean_log, std_log = rw_log_distribution(S0, h, params)
                for q in quantiles:
                    z = norm_ppf(q)
                    qvals[f"q{int(q*100):02d}"] = exp(mean_log + z * std_log)

            else:
                raise ValueError("method must be 'normal' or 'bootstrap'")

        rows.append(
            {
                "current_date": start_obs_date.date().isoformat(),
                "target_date": target_obs_date.date().isoformat(),
                "h_steps": h,
                "S0": S0,
                "window_mean": window_mean,
                "window_std": window_std,
                "method": method,
                **qvals,
                "actual": actual,
            }
        )

    return pd.DataFrame(rows)