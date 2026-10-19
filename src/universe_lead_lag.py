"""Universo reducido para lead-lag: BTC + top alts líquidas."""
from __future__ import annotations

# Para lead-lag necesitamos alts con liquidez alta en 1m (slippage bajo).
# Subconjunto del universo general.
LEAD_LAG_UNIVERSE: list[str] = [
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "DOGEUSDT",
]
LEADER = "BTCUSDT"
