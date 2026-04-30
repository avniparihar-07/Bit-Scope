"""
BTC 1h Range Forecaster — Core Model
FIGARCH(1,d,1) + Cyber-GBM with Student-t, rolling entropy, volatility clustering.
Directly adapted from the original Colab (USDCHF FIGARCH) -> BTCUSDT 1h Binance bars.
"""

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model


# ── Binance data ──────────────────────────────────────────────────────────────

def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch up to `limit` closed hourly bars from Binance public API.
    No API key needed — fully public endpoint.
    Returns DataFrame with: open_time, open, high, low, close, volume
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json(), columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Drop the last bar — it may still be forming (not yet closed)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


# ── Rolling entropy (from original Colab) ────────────────────────────────────

def rolling_entropy(x: pd.Series, window: int = 60, bins: int = 20) -> pd.Series:
    """
    Rolling Shannon entropy of residuals distribution.
    High entropy -> chaotic/crisis-like regime -> wider range prediction.
    """
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    return x.rolling(window).apply(ent, raw=True)


# ── FIGARCH fit ───────────────────────────────────────────────────────────────

def fit_figarch(log_ret: pd.Series):
    """
    Fit FIGARCH(1,d,1) with Student-t innovations.
    Identical to original Colab. BTC returns * 100 for numerical stability.
    Returns the fitted arch model result object.
    """
    am = arch_model(log_ret * 100, vol="FIGARCH", p=1, o=0, q=1, dist="studentst")
    res = am.fit(disp="off", show_warning=False)
    return res


# ── Cyber-GBM helpers (from original Colab) ───────────────────────────────────

def build_cyber_params(H_series: pd.Series, M_series: pd.Series) -> dict:
    """
    Build base Cyber-GBM parameters ensuring stability:
    alpha * H_max + delta * M_max < 1
    """
    H_max = max(float(H_series.max()), 1e-9)
    M_max = max(float(M_series.max()), 1e-9)
    alpha0, delta0 = 0.5, 0.3
    if alpha0 * H_max + delta0 * M_max >= 1:
        fac = 0.95 / (alpha0 * H_max + delta0 * M_max)
        alpha0 *= fac
        delta0 *= fac
    return {"alpha": alpha0, "delta": delta0, "gamma": 0.2, "kappa": 0.1, "eta": 1e-3}


def update_params(p: dict, sigma2: float, bar_sigma2: float, t: int) -> dict:
    """Adaptive update of mean-reversion speed gamma (from original Colab)."""
    err = sigma2 - bar_sigma2
    lr = p["eta"] / (1 + t ** 0.55)
    p["gamma"] = float(np.clip(p["gamma"] + lr * err, 0.01, 0.5))
    return p


# ── Single Cyber-GBM path ─────────────────────────────────────────────────────

def simulate_cyber_gbm_path(
    S0: float,
    mu: float,
    sigma_fig: pd.Series,
    H: pd.Series,
    M: pd.Series,
    redundancy: pd.Series,
    info_filter: pd.Series,
    params: dict,
    bar_sigma2: float,
    n_steps: int,
    nu: float,
    dt: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Simulate one Cyber-GBM path n_steps ahead.
    Structure identical to original Colab's simulate_cyber_gbm().
    """
    S = np.zeros(n_steps + 1)
    S[0] = S0
    sigma2 = float(sigma_fig.iloc[-1] ** 2)

    H_max = max(float(H.max()), 1e-9)
    M_max = max(float(M.max()), 1e-9)

    for t in range(1, n_steps + 1):
        H_val = min(float(H.iloc[-1]) / H_max, 1.0)
        M_val = min(float(M.iloc[-1]) / M_max, 1.0)

        crisis = (H_val > 0.8) or (M_val > 0.8)
        delta_t = params["delta"] if crisis else 0.0

        sigma2 = (
            float(sigma_fig.iloc[-1]) ** 2
            * (1 + params["alpha"] * H_val + delta_t * M_val)
            + params["gamma"] * (bar_sigma2 - sigma2)
        )
        sigma2 *= max(1e-12, float(redundancy.iloc[-1]))
        sigma2 *= 1.0 + 0.5 * float(info_filter.iloc[-1])
        sigma2 = max(eps, min(sigma2, 0.5))

        # Student-t shock with correct variance normalisation
        Z = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)

        params = update_params(params, sigma2, bar_sigma2, t)

    return S


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def simulate_mc(
    S0: float,
    mu: float,
    sigma_fig: pd.Series,
    H: pd.Series,
    M: pd.Series,
    redundancy: pd.Series,
    info_filter: pd.Series,
    bar_sigma2: float,
    nu: float,
    n_sims: int = 10_000,
    n_steps: int = 1,
) -> np.ndarray:
    """
    Run n_sims Cyber-GBM paths each n_steps ahead.
    Returns array (n_sims, n_steps+1).
    """
    base_params = build_cyber_params(H, M)
    out = np.zeros((n_sims, n_steps + 1))
    for i in range(n_sims):
        out[i] = simulate_cyber_gbm_path(
            S0, mu, sigma_fig, H, M, redundancy, info_filter,
            base_params.copy(), bar_sigma2, n_steps, nu,
        )
    return out


# ── Main predict function (no-peek safe) ──────────────────────────────────────

def predict_range(
    close_prices: np.ndarray,
    lookback: int = 500,
    n_sims: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    """
    Predict the 95% price range for the NEXT 1h bar.

    NO future data may be passed in close_prices — caller is responsible
    for slicing closes[:i] when backtesting (no-peek guarantee).

    Parameters
    ----------
    close_prices : np.ndarray, chronological, up to and including current bar
    lookback     : bars used to fit FIGARCH (500 recommended for BTC)
    n_sims       : Monte Carlo paths
    confidence   : interval level (0.95 -> 2.5th-97.5th percentile)

    Returns
    -------
    dict: lower, upper, mid, width, current_price, nu, sigma_last, H_last
    """
    # Use at most `lookback` bars of price history
    prices_slice = pd.Series(close_prices[-(lookback + 1):])
    log_ret = np.log(prices_slice / prices_slice.shift(1)).dropna()

    S0 = float(close_prices[-1])
    mu = float(log_ret.mean())

    # ── Fit FIGARCH(1,d,1) with Student-t ─────────────────────────────────────
    res = fit_figarch(log_ret)
    sigma_fig = res.conditional_volatility / 100.0   # rescale back from ×100

    # ── Degrees of freedom from standardised residuals ────────────────────────
    resid = (log_ret * 100 - res.params["mu"]) / res.conditional_volatility
    nu = float(max(4.0, stats.t.fit(resid, floc=0, fscale=1)[0]))

    # ── Entropy & magnitude regime filters ────────────────────────────────────
    H_series = rolling_entropy(resid).fillna(0.0)
    M_series = log_ret.abs().rolling(60).mean().fillna(log_ret.abs().mean())

    # Redundancy: short-term / medium-term variance ratio (stays near 1)
    redundancy = (
        1.0 + 0.1 * np.log1p(
            prices_slice.rolling(5).var() / prices_slice.rolling(20).var()
        )
    ).fillna(1.0)

    info_filter = (H_series > H_series.mean()).astype(float)

    bar_sigma2 = float((sigma_fig ** 2).mean())

    # ── 1-step-ahead Monte Carlo ───────────────────────────────────────────────
    paths = simulate_mc(
        S0, mu, sigma_fig, H_series, M_series,
        redundancy, info_filter,
        bar_sigma2, nu,
        n_sims=n_sims,
        n_steps=1,
    )

    S_t1 = paths[:, 1]
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(S_t1, alpha * 100))
    upper = float(np.percentile(S_t1, (1.0 - alpha) * 100))

    return {
        "lower":         lower,
        "upper":         upper,
        "mid":           (lower + upper) / 2.0,
        "width":         upper - lower,
        "current_price": S0,
        "nu":            nu,
        "sigma_last":    float(sigma_fig.iloc[-1]),
        "H_last":        float(H_series.iloc[-1]),
    }


# ── Evaluation (mirrors Colab evaluate()) ────────────────────────────────────

def winkler_score(lower: float, upper: float, actual: float, alpha: float = 0.05) -> float:
    """Winkler interval score. Lower = better forecaster."""
    width = upper - lower
    if actual < lower:
        return width + (2 / alpha) * (lower - actual)
    elif actual > upper:
        return width + (2 / alpha) * (actual - upper)
    return width


def evaluate(predictions: list) -> dict:
    """
    predictions: list of dicts with keys lower, upper, actual
    Returns coverage, avg_width, avg_winkler, n
    """
    covered, widths, winklers = [], [], []
    for p in predictions:
        lo, hi, act = p["lower"], p["upper"], p["actual"]
        covered.append(lo <= act <= hi)
        widths.append(hi - lo)
        winklers.append(winkler_score(lo, hi, act))
    return {
        "coverage":    float(np.mean(covered)),
        "avg_width":   float(np.mean(widths)),
        "avg_winkler": float(np.mean(winklers)),
        "n":           len(predictions),
    }
