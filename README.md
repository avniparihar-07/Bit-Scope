# BTC 1-Hour Range Forecaster

**FIGARCH(1,d,1) + Cyber-GBM** Monte Carlo forecaster for the BTCUSDT hourly bar.
Directly adapted from the original Colab notebook (USDCHF FIGARCH model).

## Files

```
├── model.py        Core: Binance fetcher, FIGARCH fit, Cyber-GBM, evaluate()
├── backtest.py     Part A: 30-day walk-forward backtest → backtest_results.jsonl
├── dashboard.py    Part B+C: Streamlit live dashboard
├── requirements.txt
└── README.md
```

## Quick start (local)

```bash
pip install -r requirements.txt

# Part A — 30-day backtest (~25-40 min on CPU)
python backtest.py

# Part B/C — live dashboard
streamlit run dashboard.py
```

## Deploy to Streamlit Community Cloud (free, ~60s)

1. Push this folder to a **public** GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo / branch `main` / file `dashboard.py`
4. Click **Deploy** → copy the public URL into your submission

## How the model works

### From original Colab → adapted for BTC 1h

| Original Colab | This project |
|----------------|-------------|
| EODHD API (paid) | Binance public API (free, no key) |
| USDCHF daily bars | BTCUSDT 1h bars |
| Daily returns ×100 | Hourly returns ×100 (same FIGARCH scale trick) |
| Backtest: 252 days | Backtest: ~720 hours (30 days) |

### FIGARCH(1,d,1)
Fractionally Integrated GARCH captures **long memory in volatility** — BTC's
volatility clusters persist longer than standard GARCH predicts. The `d`
parameter (0 < d < 0.5) controls the fractional integration degree.

### Cyber-GBM
On top of FIGARCH volatility, the model applies two regime filters:
- **Entropy H**: Rolling Shannon entropy of standardised residuals. High H → 
  chaotic regime → alpha and delta terms widen σ².
- **Magnitude M**: Rolling mean of |returns|. High M → trending volatility.
- **Crisis mode**: When H > 80th percentile OR M > 80th percentile, the delta
  term activates and widens the predicted range significantly.
- **Redundancy**: Short-term (5-bar) vs medium-term (20-bar) variance ratio
  scales σ² upward when recent price moves are unusually large.

### No-peek guarantee
In `backtest.py`, bar `i`'s prediction uses `closes[:i]` — a Python slice that
physically cannot include `closes[i]`. The actual price `closes[i]` is only
accessed afterwards for scoring.

### Fat tails
`scipy.stats.t.fit(residuals)` estimates degrees-of-freedom ν from the data.
BTC typically yields ν ≈ 3–6, giving the simulation far more probability mass
in extreme moves than a normal distribution would.

## Scoring targets

| Metric | Target |
|--------|--------|
| Coverage | ≈ 0.95 |
| Avg Width | Narrow as possible while hitting coverage |
| Avg Winkler | Lower = better |
