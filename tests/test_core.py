import time

import numpy as np
import pandas as pd
import pytest

from trader.agents.base import Context
from trader.agents.execution import ExecutionAgent
from trader.agents.intel import fuse
from trader.agents.news import aggregate, score_text, tag_symbols
from trader.agents.risk import RiskAgent
from trader.backtest import resample, run_backtest
from trader.broker import PaperBroker
from trader.config import load_config
from trader.gate import check_gate
from trader.indicators import add_htf, add_indicators
from trader.journal import Journal
from trader.logic import check_exit, plan_entries, size_position, strategy_weights, update_trailing
from trader.strategies import Signal, detect_regime


@pytest.fixture
def cfg():
    return load_config(None, {"db_path": ":memory:"})


@pytest.fixture
def journal():
    return Journal(":memory:")


def synthetic(n=1500, seed=1, drift=0.0004):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.006, n))
    low = close * (1 - rng.uniform(0, 0.006, n))
    open_ = np.r_[close[0], close[:-1]]
    idx = (np.arange(n) * 3_600_000 + 1_700_000_000_000)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def test_resample_keeps_ms_timestamps():
    df = synthetic(48)
    r = resample(df, "4h")
    assert r.index[0] == df.index[0] // 14_400_000 * 14_400_000
    assert r["high"].max() == df["high"].max()


def test_indicators_no_lookahead():
    df = synthetic(600)
    full = add_indicators(df)
    part = add_indicators(df.iloc[:400])
    for col in ("ema20", "rsi14", "atr14", "dc_high20", "bb_low"):
        assert np.isclose(full[col].iloc[399], part[col].iloc[399], equal_nan=True)


def test_regime_detection():
    row = {"close": 110, "ema20": 108, "ema50": 105, "ema100": 100, "adx14": 30, "atr_rank": 0.5, "dd_long": -0.02}
    assert detect_regime(row) == "trend_up"
    row.update(atr_rank=0.95)
    assert detect_regime(row) == "high_vol"
    row.update(dd_long=-0.3, close=90)
    assert detect_regime(row) == "crisis"
    assert detect_regime({"close": 1}) == "unknown"
    row = {"close": 110, "ema20": 108, "ema50": 105, "ema100": 100, "adx14": 30, "atr_rank": 0.5, "dd_long": -0.02}
    assert detect_regime({**row, "htf_ok": 0.0}) == "bear"
    assert detect_regime({**row, "htf_ok": float("nan")}) == "unknown"
    assert detect_regime({**row, "htf_ok": 1.0}) == "trend_up"


def test_htf_filter_uses_only_completed_days():
    day = 86_400_000
    idx = np.arange(30) * day + 19_700 * day
    daily = pd.DataFrame({"close": np.r_[np.arange(1.0, 26.0), np.ones(5)]}, index=idx)
    hourly = pd.DataFrame({"close": 1.0}, index=np.arange(20 * 24, 30 * 24) * 3_600_000 + idx[0])
    out = add_htf(hourly, daily, 20)
    assert out.loc[idx[25] + 12 * 3_600_000, "htf_ok"] == 1.0  # crash day 25 has not closed yet
    assert out.loc[idx[26] + 3_600_000, "htf_ok"] == 0.0  # day 25 closed below its SMA
    assert np.isnan(add_htf(hourly, daily.iloc[:10], 20)["htf_ok"]).all()


def sig(**kw):
    base = dict(symbol="BTC/USD", strategy="trend", regime="trend_up", price=100.0, stop=97.0, target=110.0,
                strength=1.0, trail_atr=3.0)
    base.update(kw)
    return Signal(**base)


def test_sizing_never_exceeds_max_risk(cfg):
    r = {**cfg.risk, "allocation": "risk"}
    d = size_position(sig(intel_mult=1.25), 1000, 1000, r, 0.25)
    assert d.reason == "ok"
    assert d.risk_amount <= 1000 * r["max_risk_per_trade"] + 1e-9
    d2 = size_position(sig(intel_mult=0.5), 1000, 1000, r, 0.25)
    assert d2.risk_amount < d.risk_amount


def test_sizing_caps_and_fee_filter(cfg):
    r = {**cfg.risk, "allocation": "risk"}
    d = size_position(sig(stop=99.9), 10, 10, r, 0.25)
    assert d.cost <= 10 * r["max_position_fraction"] + 1e-9
    assert size_position(sig(target=100.5), 10, 10, r, 0.25).reason.startswith("reward")
    assert "minimum" in size_position(sig(), 10, 10, r, 0.25, min_amount=1.0).reason


def test_full_allocation_commits_all_cash(cfg):
    r = dict(cfg.risk)
    assert r["allocation"] == "full"
    d = size_position(sig(), 1000, 1000, r, 0.25)
    assert d.reason == "ok"
    assert d.cost + d.cost * r["fee_rate"] <= 1000
    assert d.cost > 0.98 * 1000 / (1 + r["fee_rate"])  # essentially everything
    half = size_position(sig(), 1000, 1000, r, 0.25, split=2)
    assert half.cost == pytest.approx(d.cost / 2)
    assert "fully invested" in size_position(sig(), 1000, 5, r, 0.25).reason  # dust stays put
    assert size_position(sig(target=100.5), 1000, 1000, r, 0.25).reason.startswith("reward")


def test_plan_entries_one_per_symbol_and_skips_blocked():
    a = sig(symbol="BTC/USD", strategy="trend")
    b = sig(symbol="BTC/USD", strategy="momentum")
    c = sig(symbol="ETH/USD", strategy="trend")
    d = sig(symbol="SOL/USD", strategy="meanrev")
    assert plan_entries([a, b, c, d], set(), 2) == [a, c]
    assert plan_entries([a, b, c, d], {"ETH/USD"}, 2) == [a, d]
    assert plan_entries([a, b, c, d], set(), 2, {("BTC/USD", "trend")}) == [b, c]


def test_trailing_and_exits():
    t = {"strategy": "trend", "entry_price": 100, "stop": 97, "target": 110, "trail_atr": 3, "highest": 100}
    t.update(update_trailing(t, 105))
    assert t["stop"] == 102 and t["highest"] == 105
    assert check_exit(t, 101, 104, 103, None, None)[0] == "stop"
    mr = {"strategy": "meanrev", "entry_price": 100, "stop": 97, "target": 103, "trail_atr": None}
    assert check_exit(mr, 99, 104, 103, None, None) == ("target", 103)
    assert check_exit(mr, 96, 104, 103, None, None)[0] == "stop"  # both touched -> assume stop


def test_strategy_weights():
    trades = [{"strategy": "trend", "pnl": 1.0}] * 8 + [{"strategy": "trend", "pnl": -0.5}] * 2
    trades += [{"strategy": "meanrev", "pnl": -1.0}] * 6
    w = strategy_weights(trades)
    assert w["trend"] == 1.5 and w["meanrev"] == 0.5


def test_news_scoring_and_tags():
    assert score_text("Exchange hacked, $200M drained") < -0.4
    assert score_text("Bitcoin ETF approval sparks rally") > 0.4
    assert tag_symbols("Ethereum upgrade goes live", ["BTC", "ETH"]) == ["ETH"]
    assert "MARKET" in tag_symbols("Fed holds rates steady", ["BTC"])
    now = time.time()
    rows = [{"published": now, "score": -0.8, "symbols": ["BTC"]} for _ in range(6)]
    agg = aggregate(rows, now, 6, ["BTC", "ETH"])
    assert agg["BTC"]["score"] < -0.7 and agg["ETH"]["count"] == 0


def test_intel_fusion_bounds(cfg):
    ic = dict(cfg.intel)
    good = fuse("BTC/USD", {"BTC": {"score": 1.0, "count": 5}, "MARKET": {"score": 1.0, "count": 5}}, None, {}, ic)
    assert good["mult"] == pytest.approx(1 + ic["max_size_boost"]) and good["veto"] is None
    bad = fuse("BTC/USD", {"BTC": {"score": -0.6, "count": 5}, "MARKET": {"score": 0.0, "count": 5}}, None, {}, ic)
    assert bad["veto"] and bad["mult"] < 1
    greedy = fuse("BTC/USD", {}, None, {"fear_greed": {"value": 90}}, ic)
    assert greedy["mult"] == pytest.approx(0.8)


def test_paper_round_trip_and_kill_switch(cfg, journal):
    broker = PaperBroker(cfg, journal)
    ex = ExecutionAgent(cfg, journal, broker)
    ctx = Context(now=time.time(), prices={"BTC/USD": 100.0}, signals=[sig(stop=98.0, target=106.0)])
    ex.run(ctx)
    assert len(journal.open_trades("paper")) == 1
    risk = RiskAgent(cfg, journal, broker)
    risk.run(Context(now=time.time(), prices={"BTC/USD": 100.0}))
    crash = Context(now=time.time(), prices={"BTC/USD": 1.0})
    risk.run(crash)
    assert crash.flatten and journal.get_state("halted")
    ex.run(crash)
    assert not journal.open_trades("paper")
    assert journal.closed_trades("paper")[0]["exit_reason"] == "flatten"


def test_target_stops_loop(cfg, journal):
    broker = PaperBroker(cfg, journal)
    journal.set_state("paper_cash", 150.0)
    ctx = Context(now=time.time(), prices={})
    RiskAgent(cfg, journal, broker).run(ctx)
    assert ctx.stop_loop and journal.get_state("goal_reached")


def test_gate_locked_without_trades(cfg, journal):
    ok, stats, problems = check_gate(cfg, journal)
    assert not ok and stats["trades"] == 0 and problems


def test_backtest_smoke(cfg):
    data = {"BTC/USD": synthetic(1500, 1), "ETH/USD": synthetic(1500, 2)}
    res = run_backtest(cfg, data, 10.0)
    assert res["final"] > 0 and len(res["curve"]) > 0
    assert res["max_drawdown"] <= cfg.risk["kill_switch_drawdown"] + 0.1
