"""Replay recorded Polymarket books (from polybot.recorder) as a BookSource.

REAL books are used for windows that have a snapshot taken within `max_age` seconds before the
decision time. Other windows are skipped (mode 'only') or priced by the MODEL fallback ('mixed').
The report tags the counts of REAL vs MODEL book uses.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np

from ..models import Book
from .market_model import AssumedMarket, BookSource, TickContext


def load_snapshots(path: str | Path) -> dict[int, tuple[list[float], list[tuple[Book, Book]]]]:
    by_ws: dict[int, list] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            up = Book(float(r["up_bid"]), float(r["up_ask"]), float(r.get("up_size") or 0.0))
            down = Book(float(r["down_bid"]), float(r["down_ask"]), float(r.get("down_size") or 0.0))
            by_ws.setdefault(int(r["window_start"]), []).append((float(r["ts"]), up, down))
        except (ValueError, KeyError, TypeError):
            continue  # skip malformed lines
    out = {}
    for ws, rows in by_ws.items():
        rows.sort(key=lambda x: x[0])
        out[ws] = ([x[0] for x in rows], [(x[1], x[2]) for x in rows])
    return out


class RecordedBooks(BookSource):
    def __init__(self, path: str | Path, mode: str = "only", fallback: AssumedMarket | None = None,
                 max_age: float = 30.0):
        if mode not in ("only", "mixed"):
            raise ValueError("books mode must be 'only' or 'mixed'")
        if mode == "mixed" and fallback is None:
            raise ValueError("mixed mode needs a fallback AssumedMarket")
        self.snaps = load_snapshots(path)
        self.mode, self.fallback, self.max_age = mode, fallback, max_age
        self.lag_minutes = fallback.lag_minutes if fallback else 0
        self.real_used = self.model_used = self.skipped = 0

    def prepare(self, ws: np.ndarray) -> None:
        if self.fallback:
            self.fallback.prepare(ws)

    def books(self, ctx: TickContext):
        rec = self.snaps.get(ctx.ws)
        if rec:
            times, pairs = rec
            i = bisect.bisect_right(times, ctx.now) - 1
            if i >= 0 and ctx.now - times[i] <= self.max_age:
                self.real_used += 1
                return pairs[i]
        if self.mode == "mixed":
            b = self.fallback.books(ctx)
            if b is not None:
                self.model_used += 1
            return b
        self.skipped += 1
        return None

    def describe(self) -> dict:
        return {"source": f"RecordedBooks mode={self.mode}: REAL books where recorded, "
                          f"{'MODEL elsewhere' if self.mode == 'mixed' else 'other windows skipped'}",
                "windows_with_recordings": len(self.snaps), "real_book_ticks": self.real_used,
                "model_book_ticks": self.model_used, "skipped_ticks": self.skipped,
                "fallback": self.fallback.describe() if self.fallback else None}
