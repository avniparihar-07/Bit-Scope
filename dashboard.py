"""
Part B + C — Live BTC Forecaster Dashboard (Streamlit)
FIGARCH(1,d,1) + Cyber-GBM · Student-t · Rolling entropy regime detection

Deploy:
  1. Push repo to GitHub (public)
  2. share.streamlit.io -> New app -> dashboard.py
  3. Free public URL in ~60 seconds

Run locally:
  streamlit run dashboard.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import fetch_binance_klines, predict_range, evaluate

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC 1h Forecaster",
    page_icon="₿",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
LOOKBACK       = 500      # bars for FIGARCH fit
N_BARS_FETCH   = 600      # fetch a bit more than lookback
N_SIMS         = 10_000
CONFIDENCE     = 0.95
BACKTEST_FILE  = "backtest_results.jsonl"
HISTORY_FILE   = "live_history.jsonl"    # Part C persistence


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 min at most
def load_live_data() -> pd.DataFrame:
    return fetch_binance_klines(limit=N_BARS_FETCH)


@st.cache_data(ttl=3600)  # re-run model at most once per hour
def run_model(closes_tuple: tuple) -> dict:
    """Cached FIGARCH model run — key is a tuple of the last few closes."""
    closes = np.array(closes_tuple)
    return predict_range(closes, lookback=LOOKBACK, n_sims=N_SIMS, confidence=CONFIDENCE)


def load_backtest_metrics():
    if not Path(BACKTEST_FILE).exists():
        return None
    records = []
    with open(BACKTEST_FILE) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return evaluate(records) if records else None


# ── Part C persistence ────────────────────────────────────────────────────────

def load_live_history() -> list:
    if not Path(HISTORY_FILE).exists():
        return []
    out = []
    with open(HISTORY_FILE) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def save_prediction(bar_ts: str, record: dict):
    """Append only if this bar_ts isn't already saved."""
    history = load_live_history()
    known = {h["bar_ts"] for h in history}
    if bar_ts not in known:
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("₿  BTC/USDT · 1-Hour Range Forecaster")
    st.caption(
        "FIGARCH(1,d,1) + Cyber-GBM · Student-t fat tails · "
        "Rolling entropy regime detection · 95% confidence interval · Live from Binance"
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Fetching latest BTCUSDT 1h bars from Binance…"):
        try:
            df = load_live_data()
        except Exception as e:
            st.error(f"❌ Failed to fetch Binance data: {e}")
            st.stop()

    # ── Run FIGARCH model ──────────────────────────────────────────────────────
    with st.spinner("Running FIGARCH + Cyber-GBM Monte Carlo (this takes ~10s)…"):
        # Use last LOOKBACK+1 closes as cache key (tuple is hashable)
        closes = df["close"].values
        cache_key = tuple(closes[-(LOOKBACK + 1):])
        try:
            pred = run_model(cache_key)
        except Exception as e:
            st.error(f"❌ Model error: {e}")
            st.stop()

    current_price = closes[-1]
    prev_price    = closes[-2]
    change_pct    = (current_price - prev_price) / prev_price * 100

    # ── Part C: persist this prediction ───────────────────────────────────────
    bar_ts = str(df["open_time"].iloc[-1])
    save_prediction(bar_ts, {
        "bar_ts":        bar_ts,
        "fetched_utc":   datetime.now(timezone.utc).isoformat(),
        "current_price": current_price,
        "lower":         pred["lower"],
        "upper":         pred["upper"],
        "width":         pred["width"],
        "nu":            pred["nu"],
        "sigma_last":    pred["sigma_last"],
        "H_last":        pred["H_last"],
    })
    history = load_live_history()

    # ── Load backtest metrics ──────────────────────────────────────────────────
    bt = load_backtest_metrics()

    # ═════════════════════════════════════════════════════════════════════════
    # Row 1: Key metrics
    # ═════════════════════════════════════════════════════════════════════════
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BTC Price",            f"${current_price:,.2f}", f"{change_pct:+.2f}%")
    c2.metric("Predicted Low (95%)",  f"${pred['lower']:,.2f}")
    c3.metric("Predicted High (95%)", f"${pred['upper']:,.2f}")
    c4.metric("Range Width",          f"${pred['width']:,.2f}")
    if bt:
        c5.metric("Backtest Coverage", f"{bt['coverage']:.1%}",
                  help="Fraction of 720 hourly backtests inside the predicted 95% range")
    else:
        c5.metric("Coverage", "Run backtest.py",
                  help="Execute backtest.py to populate this metric")

    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # Row 2: Backtest scorecard
    # ═════════════════════════════════════════════════════════════════════════
    if bt:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Coverage",       f"{bt['coverage']:.4f}", help="Target ≈ 0.9500")
        b2.metric("Avg Width",      f"${bt['avg_width']:,.0f}")
        b3.metric("Avg Winkler",    f"{bt['avg_winkler']:,.1f}", help="Lower = better")
        b4.metric("Bars Backtested",f"{bt['n']:,}")
        st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # Row 3: Model diagnostics
    # ═════════════════════════════════════════════════════════════════════════
    with st.expander("🔬 Model diagnostics", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Student-t df (ν)",   f"{pred['nu']:.2f}",
                  help="Lower df → heavier tails → wider range. BTC typically 3–6.")
        d2.metric("FIGARCH σ (last bar)",f"{pred['sigma_last']*100:.4f}%",
                  help="Conditional volatility of the last bar")
        d3.metric("Entropy H (regime)", f"{pred['H_last']:.3f}",
                  help="High entropy → crisis/chaotic regime detected → wider range")

    # ═════════════════════════════════════════════════════════════════════════
    # Row 4: Price chart — last 50 bars + 1h ahead ribbon
    # ═════════════════════════════════════════════════════════════════════════
    chart_n  = 50
    chart_df = df.iloc[-chart_n:].copy()
    last_ts  = pd.Timestamp(chart_df["open_time"].iloc[-1])
    next_ts  = last_ts + pd.Timedelta(hours=1)

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=chart_df["open_time"],
        open=chart_df["open"], high=chart_df["high"],
        low=chart_df["low"],   close=chart_df["close"],
        name="OHLC",
        increasing_line_color="#00c896",
        decreasing_line_color="#ff4b6e",
    ))

    # Predicted range ribbon (shaded area for the next bar)
    fig.add_trace(go.Scatter(
        x=[last_ts, next_ts, next_ts, last_ts],
        y=[pred["lower"], pred["lower"], pred["upper"], pred["upper"]],
        fill="toself",
        fillcolor="rgba(88, 130, 255, 0.22)",
        line=dict(color="rgba(88, 130, 255, 0.85)", width=1.5, dash="dot"),
        name="95% Predicted Range",
    ))

    # Boundary labels
    for y_val, label in [
        (pred["lower"], f"Low  ${pred['lower']:,.0f}"),
        (pred["upper"], f"High ${pred['upper']:,.0f}"),
    ]:
        fig.add_hline(
            y=y_val, line_dash="dash", line_color="rgba(88,130,255,0.8)",
            annotation_text=label, annotation_position="right",
            annotation_font_color="rgba(88,130,255,1)",
        )

    fig.update_layout(
        title=f"BTCUSDT — Last {chart_n} bars  +  1h Ahead Forecast",
        xaxis_title="Time (UTC)",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        height=540,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=10, r=90, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # Row 5: Part C — growing prediction history table
    # ═════════════════════════════════════════════════════════════════════════
    if history:
        st.subheader(f"📋 Live Prediction History  ({len(history)} saved)")
        hdf = pd.DataFrame(history[::-1])   # newest first
        hdf["bar_ts"] = pd.to_datetime(hdf["bar_ts"]).dt.strftime("%Y-%m-%d %H:%M UTC")
        for col in ["current_price", "lower", "upper", "width"]:
            hdf[col] = hdf[col].map("${:,.2f}".format)
        hdf = hdf.rename(columns={
            "bar_ts":        "Bar Time",
            "current_price": "Price at Prediction",
            "lower":         "Predicted Low",
            "upper":         "Predicted High",
            "width":         "Range Width",
        })
        show_cols = ["Bar Time", "Price at Prediction", "Predicted Low",
                     "Predicted High", "Range Width"]
        st.dataframe(hdf[show_cols], use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # Footer
    # ═════════════════════════════════════════════════════════════════════════
    st.caption(
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ·  "
        f"Model: FIGARCH(1,d,1) + Cyber-GBM  ·  "
        f"ν={pred['nu']:.1f}  ·  Lookback={LOOKBACK} bars  ·  "
        f"MC paths={N_SIMS:,}  ·  Data: Binance BTCUSDT 1h"
    )


if __name__ == "__main__":
    main()
