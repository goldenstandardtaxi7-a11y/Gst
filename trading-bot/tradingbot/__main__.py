import argparse
import logging
import os
import sys

from .backtest import buy_and_hold, compute_stats, run_backtest
from .config import StrategyConfig, broker_config_from_env, load_env_file
from .data import load_csv_dir, synthetic_closes

STATE_PATH = os.path.join("state", "state.json")


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join("logs", "bot.log"))])


def print_stats(name, stats):
    print(f"\n{name}")
    for k, v in stats.items():
        print(f"  {k:<14} {v:.2%}" if k not in ("years", "sharpe") else f"  {k:<14} {v:.2f}")


def cmd_backtest(args):
    cfg = StrategyConfig(allow_shorts=not args.no_shorts)
    if args.synthetic:
        closes = synthetic_closes(cfg.universe)
        print("WARNING: synthetic random data - shows the mechanics only, says nothing about real performance.")
    elif args.csv_dir:
        closes = load_csv_dir(args.csv_dir, cfg.universe)
    else:
        from .broker import AlpacaBroker
        closes = AlpacaBroker(broker_config_from_env()).daily_closes(cfg.universe, args.start)
    cfg.universe = list(closes.columns)
    result = run_backtest(closes, cfg, cost_bps=args.cost_bps)
    print_stats("Strategy" + (" (long only)" if args.no_shorts else " (long + short)"), result.stats)
    print(f"  {'avg turnover':<14} {result.turnover.sum() / result.stats['years']:.2f}x / year")
    bench = "SPY" if "SPY" in closes else closes.columns[0]
    print_stats(f"Buy & hold {bench}", compute_stats(buy_and_hold(closes[bench])))


def cmd_trade(args):
    from .broker import AlpacaBroker
    from .runner import State, run_cycle
    setup_logging()
    cfg = broker_config_from_env()
    logging.getLogger("tradingbot").info("Mode: %s%s", cfg.mode.upper(), " (dry run)" if args.dry_run else "")
    run_cycle(AlpacaBroker(cfg), State(STATE_PATH), dry_run=args.dry_run)


def cmd_reset_halt(_args):
    from .runner import State
    state = State(STATE_PATH)
    state.data.update(halted=False, halt_reason="", peak_equity=0.0)
    state.save()
    print("Halt cleared; peak equity will be re-measured on the next run.")


def main():
    load_env_file()
    parser = argparse.ArgumentParser(prog="tradingbot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    bt = sub.add_parser("backtest", help="test the strategy on historical data")
    bt.add_argument("--synthetic", action="store_true", help="use random demo data")
    bt.add_argument("--csv-dir", help="folder with SYMBOL.csv files")
    bt.add_argument("--start", default="2016-01-01", help="start date when downloading from Alpaca")
    bt.add_argument("--no-shorts", action="store_true", help="long-only, for comparison")
    bt.add_argument("--cost-bps", type=float, default=10.0, help="cost per trade in basis points")
    bt.set_defaults(func=cmd_backtest)
    tr = sub.add_parser("trade", help="run one trading cycle")
    tr.add_argument("--dry-run", action="store_true", help="show orders without sending them")
    tr.set_defaults(func=cmd_trade)
    sub.add_parser("reset-halt", help="clear the kill switch").set_defaults(func=cmd_reset_halt)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
