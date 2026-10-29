"""Historical BTCUSDT 1m kline loading: Binance public data download + cache, CSV, synthetic.

Real data comes from https://data.binance.vision (spot BTCUSDT 1m). Months are cached as
.npz files in the data dir, so an interrupted download resumes where it stopped.
Everything returns a `Klines` with timestamps in whole seconds (minute OPEN time).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

BASE_URL = "https://data.binance.vision/data/spot"
SYMBOL = "BTCUSDT"
Fetch = Callable[[str, Path], bool]  # (url, dest) -> True if saved, False on 404


class DataError(RuntimeError):
    pass


@dataclass
class Klines:
    ts: np.ndarray      # int64 seconds, minute open time, sorted, unique
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quality: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ts)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- parsing

def parse_csv_bytes(raw: bytes) -> tuple[np.ndarray, ...]:
    """Parse Binance kline CSV (with or without header). Returns ts_sec, o, h, l, c.

    Timestamps may be milliseconds or microseconds (spot data switched to microseconds in 2025).
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        e = np.empty(0)
        return (np.empty(0, dtype=np.int64), e, e, e, e)
    first = text.split("\n", 1)[0].split(",")[0].strip()
    try:
        float(first)
        skip = 0
    except ValueError:
        skip = 1
    arr = np.loadtxt(io.StringIO(text), delimiter=",", usecols=(0, 1, 2, 3, 4), skiprows=skip,
                     dtype=np.float64, ndmin=2)
    if arr.size == 0:
        e = np.empty(0)
        return (np.empty(0, dtype=np.int64), e, e, e, e)
    t = arr[:, 0]
    div = np.where(t > 1e14, 1e6, 1e3)  # per row: >1e14 means microseconds
    ts = np.floor(t / div).astype(np.int64)
    return ts, arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]


def parse_zip(path: Path) -> tuple[np.ndarray, ...]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise DataError(f"no csv inside {path}")
        return parse_csv_bytes(z.read(names[0]))


def clean(ts, o, h, l, c) -> Klines:
    """Sort, drop invalid rows and duplicates, and build the data quality summary."""
    ts = np.asarray(ts, dtype=np.int64)
    o, h, l, c = (np.asarray(x, dtype=np.float64) for x in (o, h, l, c))
    n_raw = len(ts)
    ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (c > 0) & (o > 0)
    n_invalid = int((~ok).sum())
    ts, o, h, l, c = ts[ok], o[ok], h[ok], l[ok], c[ok]
    order = np.argsort(ts, kind="stable")
    ts, o, h, l, c = ts[order], o[order], h[order], l[order], c[order]
    if len(ts):
        keep = np.ones(len(ts), dtype=bool)
        keep[:-1] = ts[1:] != ts[:-1]  # keep the last of equal timestamps
        n_dup = int((~keep).sum())
        ts, o, h, l, c = ts[keep], o[keep], h[keep], l[keep], c[keep]
    else:
        n_dup = 0
    q: dict = {"rows_raw": n_raw, "rows": int(len(ts)), "invalid_rows": n_invalid, "duplicates": n_dup}
    if len(ts) > 1:
        d = np.diff(ts)
        expected = int((ts[-1] - ts[0]) // 60) + 1
        gaps = d[d > 60]
        r = np.abs(np.diff(np.log(c)))
        q.update(first=int(ts[0]), last=int(ts[-1]), expected_minutes=expected,
                 missing_minutes=int(expected - len(ts)), gap_count=int(len(gaps)),
                 largest_gap_minutes=int(gaps.max() // 60 - 1) if len(gaps) else 0,
                 outliers_gt_5pct_1m=int((r > 0.05).sum()), max_abs_1m_return=float(r.max()))
    return Klines(ts, o, h, l, c, q)


# ---------------------------------------------------------------- download

def http_fetch(url: str, dest: Path, retries: int = 5, timeout: float = 60.0) -> bool:
    """Stream url to dest atomically (tmp then rename). False on 404, retries with backoff."""
    import httpx
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            os.replace(tmp, dest)
            return True
        except (httpx.HTTPError, OSError) as e:
            if attempt == retries - 1:
                raise DataError(f"download failed after {retries} tries: {url}: {e!r}") from e
            time.sleep(min(2 ** attempt, 30))
    return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _save_npz(path: Path, arrs: tuple[np.ndarray, ...]) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, ts=arrs[0], o=arrs[1], h=arrs[2], l=arrs[3], c=arrs[4])
    os.replace(tmp, path)


def _load_npz(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path) as z:
        return z["ts"], z["o"], z["h"], z["l"], z["c"]


def _fetch_and_cache(url: str, npz: Path, data_dir: Path, fetch: Fetch, verify: bool) -> bool:
    zpath = data_dir / (Path(url).name)
    try:
        if not fetch(url, zpath):
            return False
        if verify:
            cpath = data_dir / (zpath.name + ".CHECKSUM")
            try:
                have = fetch(url + ".CHECKSUM", cpath)
            except DataError:
                have = False
            if have:
                want = cpath.read_text().split()[0].strip().lower()
                if _sha256(zpath) != want:
                    raise DataError(f"checksum mismatch for {zpath.name}; delete it and retry")
                cpath.unlink(missing_ok=True)
            else:
                _log(f"  note: no checksum available for {zpath.name}, skipped verification")
        _save_npz(npz, parse_zip(zpath))
        return True
    finally:
        zpath.unlink(missing_ok=True)


def _month_start(y: int, m: int) -> dt.date:
    return dt.date(y, m, 1)


def _next_month(d: dt.date) -> dt.date:
    return dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def ensure_data(start: dt.date, end: dt.date, data_dir: str | Path, fetch: Fetch | None = None,
                verify: bool = True, today: dt.date | None = None) -> list[Path]:
    """Make sure cached .npz exist for [start, end] (inclusive dates). Returns cache files in order.

    Whole past months use monthly zips (daily zips as a fallback if the month is not published
    yet). The current month uses daily zips. Missing (404) files are skipped, not fatal.
    Already cached files are never downloaded again, so this is resumable.
    """
    fetch = fetch or http_fetch
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    today = today or dt.datetime.now(dt.timezone.utc).date()
    out: list[Path] = []
    cur = _month_start(start.year, start.month)
    while cur <= end:
        nxt = _next_month(cur)
        tag = f"{cur.year:04d}-{cur.month:02d}"
        monthly = data_dir / f"{SYMBOL}-1m-{tag}.npz"
        # a month that is over is fetched whole (load_cached clips to the requested range)
        if nxt <= today:
            if monthly.exists():
                out.append(monthly)
                cur = nxt
                continue
            url = f"{BASE_URL}/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{tag}.zip"
            _log(f"downloading {tag} (monthly)")
            if _fetch_and_cache(url, monthly, data_dir, fetch, verify):
                out.append(monthly)
                cur = nxt
                continue
            _log(f"  monthly {tag} not published, trying daily files")
        d = cur
        while d < nxt and d <= end:
            if d >= start and d < today:  # today's file is not published until tomorrow
                dn = data_dir / f"{SYMBOL}-1m-{d.isoformat()}.npz"
                if not dn.exists():
                    url = f"{BASE_URL}/daily/klines/{SYMBOL}/1m/{SYMBOL}-1m-{d.isoformat()}.zip"
                    if not _fetch_and_cache(url, dn, data_dir, fetch, verify):
                        d += dt.timedelta(days=1)
                        continue
                out.append(dn)
            d += dt.timedelta(days=1)
        cur = nxt
    return out


def load_cached(files: Iterable[Path], start_ts: int | None = None, end_ts: int | None = None) -> Klines:
    parts = [_load_npz(f) for f in files]
    if not parts:
        raise DataError("no data files available (nothing downloaded; check the date range and network)")
    cat = [np.concatenate([p[i] for p in parts]) for i in range(5)]
    ts = cat[0]
    m = np.ones(len(ts), dtype=bool)
    if start_ts is not None:
        m &= ts >= start_ts
    if end_ts is not None:
        m &= ts < end_ts
    return clean(*(a[m] for a in cat))


def load_csv(paths: Iterable[str | Path]) -> Klines:
    """Local Binance-format CSV (or zip) files."""
    parts = []
    for p in paths:
        p = Path(p)
        parts.append(parse_zip(p) if p.suffix.lower() == ".zip" else parse_csv_bytes(p.read_bytes()))
    if not parts:
        raise DataError("no csv files given")
    return clean(*(np.concatenate([p[i] for p in parts]) for i in range(5)))


def load_binance(start: dt.date, end: dt.date, data_dir: str | Path, fetch: Fetch | None = None,
                 verify: bool = True) -> Klines:
    files = ensure_data(start, end, data_dir, fetch, verify)
    s = int(dt.datetime(start.year, start.month, start.day, tzinfo=dt.timezone.utc).timestamp())
    e = int(dt.datetime(end.year, end.month, end.day, tzinfo=dt.timezone.utc).timestamp()) + 86400
    return load_cached(files, s, e)


# ---------------------------------------------------------------- synthetic

SYNTH_START = 1_577_836_800  # 2020-01-01 UTC, a multiple of 300


def synthetic(years: float = 1.0, seed: int = 0, start_price: float = 8000.0) -> Klines:
    """Deterministic fake BTC 1m bars: GBM with regime-switching vol, jumps and data gaps.

    For offline tests and smoke runs only. It has NO predictive structure, so a sane
    strategy should not make money on it (apart from fee-free noise).
    """
    rng = np.random.default_rng(seed)
    n = max(10, int(years * 365 * 1440))
    base = 0.5 / np.sqrt(365 * 1440)  # ~50% annual vol per minute
    block = 360
    nb = n // block + 1
    mult = np.array([0.5, 1.0, 2.2])
    reg = np.empty(nb, dtype=int)
    reg[0] = 1
    stay = rng.random(nb)
    pick = rng.integers(0, 3, nb)
    for i in range(1, nb):
        reg[i] = reg[i - 1] if stay[i] < 0.9 else pick[i]
    vol = np.repeat(base * mult[reg], block)[:n]
    r = rng.standard_normal(n) * vol
    jumps = rng.random(n) < 2.0 / 1440 / 30  # about 2 per month
    r += jumps * rng.standard_normal(n) * base * 12
    close = start_price * np.exp(np.cumsum(r))
    opn = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.standard_normal((2, n))) * vol * 0.5
    high = np.maximum(opn, close) * (1 + wick[0])
    low = np.minimum(opn, close) * (1 - wick[1])
    ts = SYNTH_START + 60 * np.arange(n, dtype=np.int64)
    keep = np.ones(n, dtype=bool)
    n_gaps = max(1, int(years * 6))
    for _ in range(n_gaps):  # outages of 3 to 90 minutes
        a = int(rng.integers(1000, max(1001, n - 100)))
        keep[a:a + int(rng.integers(3, 90))] = False
    return clean(ts[keep], opn[keep], high[keep], low[keep], close[keep])
