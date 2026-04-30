"""
Part A — 30-day walk-forward backtest
Run: python backtest.py

Fetches last ~750 BTCUSDT 1h bars, runs a strict no-peek backtest using
FIGARCH(1,d,1) + Cyber-GBM (same model as original Colab), writes
backtest_results.jsonl, then prints coverage / avg_width / Winkler metrics.

Runtime: ~25-40 min on CPU (FIGARCH fit per bar is the bottleneck).
Tip: reduce TEST_BARS to 100 for a quick smoke test first.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from model import fetch_binance_klines, predict_range, evaluate

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK   = 500    # bars used to fit FIGARCH each step
TEST_BARS  = 720    # ~30 days of 1h bars to backtest over
N_SIMS     = 10_000
CONFIDENCE = 0.95
OUTPUT_FILE = "backtest_results.jsonl"

# FIGARCH needs LOOKBACK bars before it; we also need 60 extra for rolling entropy
WARMUP = LOOKBACK + 65


def run_backtest():
    total_fetch = TEST_BARS + WARMUP + 10
    # Binance max per call is 1000; for >1000 you'd need multiple calls.
    # 720+565+10=1295, so we cap fetch at 1000 and adjust accordingly.
    fetch_limit = min(total_fetch, 1000)
    effective_test = fetch_limit - WARMUP - 5

    print(f"Fetching {fetch_limit} BTCUSDT 1h bars from Binance...")
    df = fetch_binance_klines(limit=fetch_limit)
    print(f"  Got {len(df)} closed bars (latest: {df['open_time'].iloc[-1]})")

    closes = df["close"].values
    times  = df["open_time"].values

    # Test window: from WARMUP to end-1 (we need closes[i+1] as actual)
    test_indices = range(WARMUP, len(closes) - 1)
    print(f"Running walk-forward backtest over {len(test_indices)} bars "
          f"(bars {WARMUP}…{len(closes)-2})...")
    print("Note: FIGARCH fit per bar takes ~2-3s. Total ~25-40 min.\n")

    predictions = []
    errors = 0
    start = time.time()

    with tqdm(test_indices, unit="bar") as pbar:
        for i in pbar:
            history = closes[:i]          # NO peeking — strictly before bar i
            actual  = float(closes[i])    # this is what we're predicting

            try:
                pred = predict_range(
                    close_prices=history,
                    lookback=LOOKBACK,
                    n_sims=N_SIMS,
                    confidence=CONFIDENCE,
                )
            except Exception as e:
                errors += 1
                pbar.set_postfix({"errors": errors, "last_err": str(e)[:30]})
                continue

            record = {
                "bar_index": i,
                "timestamp": str(times[i]),
                "lower":  pred["lower"],
                "upper":  pred["upper"],
                "mid":    pred["mid"],
                "actual": actual,
                "hit":    pred["lower"] <= actual <= pred["upper"],
                "width":  pred["width"],
                "nu":     pred["nu"],
                "sigma":  pred["sigma_last"],
            }
            predictions.append(record)

            # Running stats
            n = len(predictions)
            cov = np.mean([p["hit"] for p in predictions])
            pbar.set_postfix({
                "coverage": f"{cov:.3f}",
                "width":    f"${pred['width']:,.0f}",
                "elapsed":  f"{(time.time()-start)/60:.1f}m",
            })

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved {len(predictions)} predictions → {OUTPUT_FILE}")
    if errors:
        print(f"  ({errors} bars skipped due to fitting errors)")

    # ── Print metrics ─────────────────────────────────────────────────────────
    metrics = evaluate(predictions)
    print()
    print("=" * 52)
    print("  BACKTEST RESULTS — FIGARCH + Cyber-GBM")
    print("=" * 52)
    print(f"  Coverage     : {metrics['coverage']:.4f}   (target ≈ 0.95)")
    print(f"  Avg width    : ${metrics['avg_width']:>10,.2f}")
    print(f"  Avg Winkler  : {metrics['avg_winkler']:>12,.2f}  (lower = better)")
    print(f"  Bars tested  : {metrics['n']}")
    print("=" * 52)

    if metrics["coverage"] < 0.90:
        print("⚠  Coverage below 0.90 — model may be overconfident.")
    elif metrics["coverage"] > 0.98:
        print("⚠  Coverage above 0.98 — ranges may be too wide.")
    else:
        print("✓  Coverage is in the healthy 90–98% band!")

    elapsed = (time.time() - start) / 60
    print(f"\nTotal runtime: {elapsed:.1f} min")
    return metrics


if __name__ == "__main__":
    run_backtest()
