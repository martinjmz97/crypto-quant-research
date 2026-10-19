# Systematic crypto strategy research

A pipeline for downloading data, backtesting and falsifying strategies on Binance perpetual futures and Deribit
options. Five strategies were tested. Most did not survive, and that is the main result.

Code and comments are in Spanish with English identifiers; this README is in English.

## Results

All figures below are computed from the CSV files in `strategies/*/results/`.

| Strategy | What was tested | Result |
|---|---|---|
| **Cointegrated pairs** (`cointegration/`) | Engle–Granger style screen of 190 perpetual-futures pairs, hourly data | 55 pairs with p < 0.05, but every one has a half-life above 240 h (minimum 324 h), so the mean reversion is too slow to trade. Paused. |
| **Basis / funding** (`basis_funding/`) | Long spot + short perp (or the reverse) to collect funding, v3 includes basis P&L and a walk-forward split | 20 symbols, 912 trades. Mean Sharpe 5.3, mean annualized P&L 4.8% of position notional, max drawdown about 1.1%. Out of sample the mean falls to 2.3% annualized, with 17 of 20 symbols positive. Sound but low return. |
| **Lead–lag** (`lead_lag/`) | Cross-correlation of BTC against 10 alts on 5-minute returns, lags 1–3 | No significant lead–lag at lags 1–3 for any of the 10 alts, even though same-bar correlations are 0.49–0.82. Discarded. |
| **Cluster reversion** (`cluster_reversion/`) | Mean reversion and momentum within clusters of correlated assets | 0 of 10 symbols profitable; median expected value about −19 bps per trade after costs. Discarded. |
| **Variance risk premium** (`derivatives/`) | Short ATM straddle proxy on Deribit BTC and ETH: enter when implied vol exceeds realized vol | BTC: 77 trades, Sharpe 1.9. ETH: 67 trades, Sharpe 0.6. Exploratory only, see the caveats. |

## Caveats

- The cointegration screen is full-sample, and p-values are not corrected for testing 190 pairs, so a handful of the 55
  could be false positives. It is a screen, not a validated signal.
- The variance-risk-premium backtest is a deliberate simplification: P&L is approximated as implied vol at entry minus
  realized vol over the holding period. It does not model gamma, vega or delta hedging, and it has no out-of-sample split.
- Basis returns are relative to position notional, with no leverage applied.
- There are no automated tests in this repository yet.

## Method

- Every backtest charges fees as a per-trade cost (basis v3: 0.02% maker fee; cluster reversion: 20 bps round trip; VRP proxy: 15 bps round trip). Slippage is not modeled separately.
- Walk-forward splits and out-of-sample reporting where the strategy has parameters to fit (basis v3).
- One strategy at a time; if it needs more than about five parameters to work, assume there is no edge.
- Rule: use only information available at each point of the backtest.

## Layout

```
src/                  data loaders (Binance and Deribit public APIs, no keys), universe, metrics, statistics
strategies/<name>/    eda.py, backtest*.py and results/*.csv for each strategy
```

Stack: Python, polars, numba, NumPy/SciPy, statsmodels.

## Reproducing

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_universe            # hourly klines (public API)
python -m strategies.cointegration.screen
python -m src.download_basis_data && python -m strategies.basis_funding.backtest_v3
python -m src.download_lead_lag_data && python -m strategies.lead_lag.eda
python -m strategies.cluster_reversion.backtest
python -m src.deribit_loader && python -m strategies.derivatives.rv_calculator && python -m strategies.derivatives.backtest_vrp
```

Raw market data is not included (about 50 MB) and is git-ignored under `data/`. Results will differ slightly
depending on the date the data is downloaded.
