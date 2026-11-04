from __future__ import annotations

import argparse
import json
import sys
import threading
import time

from .config import load_config
from .journal import Journal


def _cfg(args):
    overrides = {}
    if getattr(args, "timeframe", None):
        overrides["timeframe"] = args.timeframe
    if getattr(args, "symbols", None):
        overrides["symbols"] = args.symbols.split(",")
    return load_config(args.config, overrides)


def cmd_run(args) -> int:
    from .broker import LiveBroker, PaperBroker
    from .exchange import make_exchange
    from .gate import check_gate
    from .supervisor import Supervisor

    cfg = _cfg(args)
    mode = "live" if args.live else cfg.mode
    journal = Journal(cfg.db_path)
    if mode == "live":
        ok, stats, problems = check_gate(cfg, journal)
        if not ok and not args.override_gate:
            print("Live trading is locked by the go-live gate:")
            for p in problems:
                print("  -", p)
            print("Keep paper trading, or pass --override-gate to accept the risk anyway.")
            return 2
        exchange = make_exchange(cfg.exchange, auth=True)
        broker = LiveBroker(cfg, journal, exchange)
    else:
        exchange = make_exchange(cfg.exchange)
        broker = PaperBroker(cfg, journal, exchange)
    if journal.get_state("goal_reached") and cfg.on_target == "stop":
        print("Target already reached. Raise target_equity or run `python -m trader reset-goal`.")
        return 0
    if cfg.dashboard["enabled"] and not args.no_dashboard:
        from .dashboard import serve
        threading.Thread(target=serve, args=(cfg, cfg.dashboard["host"], cfg.dashboard["port"]), daemon=True).start()
        print(f"Dashboard: http://{cfg.dashboard['host']}:{cfg.dashboard['port']}")
    Supervisor(cfg, journal, broker, exchange).run_forever(max_beats=args.beats)
    return 0


def cmd_backtest(args) -> int:
    from .backtest import fetch_history, format_result, load_limits, run_backtest

    cfg = _cfg(args)
    days = args.days or cfg.backtest["days"]
    timeframes = args.compare.split(",") if args.compare else [cfg.timeframe]
    limits = load_limits(cfg)
    for tf in timeframes:
        c = load_config(args.config, {"timeframe": tf, **({"symbols": args.symbols.split(",")} if args.symbols else {})})
        data, daily = {}, {}
        sma = int(c.htf_sma_days or 0)
        for s in c.symbols:
            src = s.replace("/" + c.quote, "/USD") if c.quote not in ("USD",) else s
            data[s] = fetch_history(c.backtest["exchange"], src, tf, days)
            if sma:
                daily[s] = fetch_history(c.backtest["exchange"], src, "1d", days + sma + 5)
        res = run_backtest(c, data, args.balance, limits, daily=daily or None)
        print(format_result(f"{','.join(c.symbols)} {tf} from ${res['start']:.0f}", res))
    print("\nBacktests exclude news/sentiment influence and assume fills at bar close. Past results do not "
          "predict future results.")
    return 0


def cmd_status(args) -> int:
    from .gate import check_gate

    cfg = _cfg(args)
    j = Journal(cfg.db_path)
    hb = j.get_state("heartbeat") or {}
    print(json.dumps({
        "last_heartbeat_age_s": round(time.time() - hb["ts"]) if hb else None,
        "mode": hb.get("mode"), "equity": hb.get("equity"), "halted": j.get_state("halted"),
        "goal_reached": j.get_state("goal_reached"), "regimes": j.get_state("regimes"),
        "intel": {k: {"mult": v["mult"], "veto": v["veto"]} for k, v in (j.get_state("intel") or {}).items()},
        "open_trades": [{k: t[k] for k in ("symbol", "strategy", "entry_price", "qty", "stop", "target")}
                        for t in j.open_trades(hb.get("mode") or cfg.mode)],
        "allocation": cfg.risk["allocation"],
        "learned_params": j.get_state("adaptive_params"),
        "cooldowns": {k: v for k, v in (j.get_state("cooldowns") or {}).items() if v > time.time()},
        "auto_resume_at": j.get_state("halt_resume_at"),
    }, indent=2, default=str))
    ok, stats, problems = check_gate(cfg, j)
    print(f"\nGo-live gate: {'PASSED' if ok else 'locked'}  {stats}")
    for p in problems:
        print("  -", p)
    return 0


def cmd_news(args) -> int:
    from .agents.base import Context
    from .agents.news import NewsAgent

    cfg = _cfg(args)
    j = Journal(cfg.db_path)
    agent = NewsAgent(cfg, j)
    print(agent.run(Context(now=time.time())))
    for r in j.query("SELECT title, source, score, symbols FROM news ORDER BY published DESC LIMIT 25"):
        print(f"  {r['score']:+.2f}  {','.join(json.loads(r['symbols'])):12} {r['title'][:90]}  [{r['source']}]")
    return 0


def cmd_resume(args) -> int:
    cfg = _cfg(args)
    j = Journal(cfg.db_path)
    j.execute("DELETE FROM state WHERE key IN ('halted', 'paper_peak', 'live_peak', 'paper_day', 'live_day', 'halt_resume_at', 'halt_handled')")
    j.log("cli", "halt cleared by user; equity peak reset")
    print("Halt cleared. The plan says: re-prove the edge on paper before going live again.")
    return 0


def cmd_reset_goal(args) -> int:
    cfg = _cfg(args)
    Journal(cfg.db_path).execute("DELETE FROM state WHERE key='goal_reached'")
    print("Goal flag cleared.")
    return 0


def cmd_reset_paper(args) -> int:
    cfg = _cfg(args)
    j = Journal(cfg.db_path)
    j.execute("DELETE FROM trades WHERE mode='paper'")
    j.execute("DELETE FROM equity WHERE mode='paper'")
    j.execute("DELETE FROM state WHERE key IN ('paper_cash','paper_peak','paper_day','halted','goal_reached','last_signal_bar',"
              "'halt_resume_at','halt_handled','learner_last_trade_paper','last_tuning_paper','cooldowns')")
    print("Paper account reset (learned parameters kept; `reset-learning` clears them).")
    return 0


def cmd_improve(args) -> int:
    """Re-tune now on recent history and (unless --dry-run) adopt the result if it is better."""
    from .learning import ADAPTIVE_KEY, current_params, effective_config, format_tuning, load_tuning_data, tune

    cfg = _cfg(args)
    j = Journal(cfg.db_path)
    eff = effective_config(cfg, j)
    print(f"Re-tuning on {cfg.learning['tune_days']} days of {cfg.timeframe} candles from {cfg.backtest['exchange']}...")
    data, daily = load_tuning_data(eff)
    res = tune(eff, data, daily, current=current_params(eff))
    print(format_tuning(res))
    print("\nTop candidates:")
    for r in res["top"]:
        p = r["params"]
        print(f"  {r['score']:+.3f}  stop {p['stop_atr_mult']} target {p['target_atr_mult']} "
              f"pos {p['max_open_positions']} {'+'.join(p['strategies'])}  val {r['val_return']:+.1%}")
    if res["adopted"] and not args.dry_run:
        j.set_state(ADAPTIVE_KEY, res["best"]["params"])
        j.log("cli", f"manual re-tune adopted {res['best']['params']}")
        print("\nAdopted. The running bot picks the new parameters up on its next heartbeat.")
    return 0


def cmd_lessons(args) -> int:
    from .learning import ADAPTIVE_KEY

    cfg = _cfg(args)
    j = Journal(cfg.db_path)
    print("Adopted parameters:", j.get_state(ADAPTIVE_KEY) or "none (config.yaml values)")
    now = time.time()
    cds = {k: v for k, v in (j.get_state("cooldowns") or {}).items() if v > now}
    print("Cool-downs:", {k: time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(v)) for k, v in cds.items()} or "none")
    print("\nRecent lessons:")
    for r in j.query("SELECT * FROM lessons ORDER BY ts DESC LIMIT 25"):
        print(f"  {time.strftime('%m-%d %H:%M', time.localtime(r['ts']))} {r['symbol']:9} {r['strategy']:8} "
              f"{r['pnl']:+.4f} {r['kind']:16} {r['lesson']}")
    print("\nRe-tunes:")
    for r in j.query("SELECT * FROM tunings ORDER BY ts DESC LIMIT 10"):
        print(f"  {time.strftime('%m-%d %H:%M', time.localtime(r['ts']))} {'ADOPTED' if r['adopted'] else 'kept   '} "
              f"score {r['current_score']:+.3f} -> {r['best_score']:+.3f}  ({r['reason']})  {r['new_params']}")
    return 0


def cmd_reset_learning(args) -> int:
    cfg = _cfg(args)
    Journal(cfg.db_path).execute("DELETE FROM state WHERE key IN ('adaptive_params','cooldowns')")
    print("Learned parameters and cool-downs cleared; config.yaml values apply again.")
    return 0


def cmd_convert(args) -> int:
    from .broker import LiveBroker
    from .exchange import make_exchange

    cfg = _cfg(args)
    ex = make_exchange(cfg.exchange, auth=True)
    broker = LiveBroker(cfg, Journal(cfg.db_path), ex)
    tickers = ex.fetch_tickers(cfg.symbols)
    prices = {s: float(t["last"]) for s, t in tickers.items() if t.get("last")}
    for line in broker.convert_holdings_to_quote(prices) or ["nothing to convert"]:
        print(line)
    return 0


def cmd_check(args) -> int:
    import sys

    import ccxt
    import pandas

    from .agents.base import Context
    from .agents.market import MarketDataAgent
    from .exchange import make_exchange
    from .strategies import detect_regime

    cfg = _cfg(args)
    print(f"python {sys.version.split()[0]}  pandas {pandas.__version__}  ccxt {ccxt.__version__}")
    agent = MarketDataAgent(cfg, Journal(":memory:"), make_exchange(cfg.exchange))
    ctx = Context(now=time.time())
    agent.run(ctx)
    days = int(cfg.htf_sma_days or 0)
    for sym, df in ctx.candles.items():
        row = df.iloc[-1].to_dict()
        daily = agent.daily.get(sym)
        n = 0 if daily is None else len(daily)
        last_d = float(daily["close"].iloc[-1]) if n else float("nan")
        sma = float(daily["close"].tail(days).mean()) if n >= days > 0 else float("nan")
        print(f"{sym}: price {ctx.prices[sym]:,.2f} | daily close {last_d:,.2f} vs {days}d avg {sma:,.2f} "
              f"({n} days, htf_ok={row.get('htf_ok')}) | ema20 {row['ema20']:,.2f} ema50 {row['ema50']:,.2f} "
              f"adx {row['adx14']:.1f} | regime {detect_regime(row)}")
    return 0


def cmd_dashboard(args) -> int:
    from .dashboard import serve

    cfg = _cfg(args)
    print(f"Dashboard: http://{cfg.dashboard['host']}:{cfg.dashboard['port']}")
    serve(cfg, cfg.dashboard["host"], cfg.dashboard["port"])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m trader", description="Autonomous multi-agent crypto trader")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start the heartbeat (paper by default)")
    r.add_argument("--live", action="store_true")
    r.add_argument("--override-gate", action="store_true", help="go live even if the paper record fails the gate")
    r.add_argument("--no-dashboard", action="store_true")
    r.add_argument("--beats", type=int, help="stop after N heartbeats (testing)")
    r.set_defaults(fn=cmd_run)
    b = sub.add_parser("backtest", help="test the strategies on historical candles")
    b.add_argument("--days", type=int)
    b.add_argument("--timeframe")
    b.add_argument("--symbols")
    b.add_argument("--balance", type=float)
    b.add_argument("--compare", help="comma list of timeframes, e.g. 15m,1h,4h")
    b.set_defaults(fn=cmd_backtest)
    im = sub.add_parser("improve", help="re-tune parameters on recent history now (the bot also does this on failures)")
    im.add_argument("--dry-run", action="store_true", help="show the result without adopting it")
    im.set_defaults(fn=cmd_improve)
    for name, fn, h in [("status", cmd_status, "show state and go-live gate"),
                        ("lessons", cmd_lessons, "show post-mortems, cool-downs, re-tunes and adopted parameters"),
                        ("reset-learning", cmd_reset_learning, "forget learned parameters and cool-downs"),
                        ("check", cmd_check, "fetch market data once and explain each regime"),
                        ("news", cmd_news, "scrape news now and print scored headlines"),
                        ("resume", cmd_resume, "clear a kill-switch halt"),
                        ("reset-goal", cmd_reset_goal, "clear the target-reached flag"),
                        ("reset-paper", cmd_reset_paper, "wipe the paper account"),
                        ("convert-deposit", cmd_convert, "live: sell deposited coins into the quote currency"),
                        ("dashboard", cmd_dashboard, "serve the dashboard only")]:
        sp = sub.add_parser(name, help=h)
        sp.set_defaults(fn=fn)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
