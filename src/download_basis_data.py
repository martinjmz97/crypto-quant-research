"""Descarga funding rates y spot prices para el universo.

Uso:
    python -m src.download_basis_data
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from .data_loader import load_funding, load_klines
from .universe import UNIVERSE

START = "2023-01-01"
END = "2024-12-31"
TIMEFRAME = "1h"


def main() -> int:
    print(f"=== Descarga funding + spot, {START}→{END} ===")
    t0 = time.time()
    for i, sym in enumerate(UNIVERSE, 1):
        ts = time.time()
        try:
            # Funding rate.
            fund = load_funding(sym, START, END, refresh=False)
            # Spot 1h.
            spot = load_klines(sym, TIMEFRAME, START, END, market="spot", refresh=False)
            print(
                f"[{i:2d}/{len(UNIVERSE)}] {sym:10s} "
                f"fund={fund.height:5d} ({fund['funding_time'].min()}→{fund['funding_time'].max()})  "
                f"spot={spot.height:5d}  "
                f"({time.time()-ts:.1f}s)"
            )
        except Exception as e:
            print(f"[{i:2d}/{len(UNIVERSE)}] {sym:10s} ERROR: {e}")
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
