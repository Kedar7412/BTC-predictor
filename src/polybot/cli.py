from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from .config import LiveNotConfirmed, load_settings, require_live_confirmation
from .engine import Engine
from .execution import LiveExecutor, PaperExecutor
from .feeds import LiveFeed, SimClock, SimulatedFeed, SystemClock
from .market import PolymarketProvider, SimulatedMarket
from .monitor import Monitor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polybot", description="Polymarket BTC 5m Up/Down bot")
    p.add_argument("--live", action="store_true", help="real orders (also needs CONFIRM_LIVE=yes)")
    p.add_argument("--dry-run", action="store_true", help="paper mode (default)")
    p.add_argument("--simulate", action="store_true", help="offline simulated feed and market")
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--config", default="config.yaml")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = load_settings(args.config)
    if args.live and args.dry_run:
        print("choose either --live or --dry-run", file=sys.stderr)
        return 2
    if args.live and args.simulate:
        print("--live cannot be combined with --simulate", file=sys.stderr)
        return 2
    try:
        s.live = require_live_confirmation(args.live)
    except LiveNotConfirmed as e:
        print(f"refusing to trade live: {e}", file=sys.stderr)
        return 2

    if args.simulate:
        clock = SimClock(start=(1_750_000_000 // 300) * 300)
        feed = SimulatedFeed(clock, sigma=s.default_sigma_per_sqrt_s)
        provider = SimulatedMarket(feed, clock, s.window_seconds, sigma=s.default_sigma_per_sqrt_s)
        s.state_path = s.trade_log_path = ""
        monitor = Monitor(None, None)
        tick_s = s.tick_seconds
        s.tick_seconds = max(tick_s, 15.0)
    else:
        clock, feed, provider = SystemClock(), LiveFeed(), PolymarketProvider(window=s.window_seconds)
        monitor = Monitor(s.trade_log_path, s.state_path)
    try:
        executor = LiveExecutor(s.bankroll, s, args.live) if s.live else PaperExecutor(s.bankroll)
    except (LiveNotConfirmed, ImportError) as e:
        print(f"cannot start live executor: {e}", file=sys.stderr)
        return 2
    logging.getLogger("polybot").info("mode=%s", "LIVE" if s.live else "PAPER")
    engine = Engine(s, feed, provider, executor, monitor, clock)
    engine.restore()
    try:
        asyncio.run(engine.run(args.max_cycles))
    except KeyboardInterrupt:
        pass
    return 0
