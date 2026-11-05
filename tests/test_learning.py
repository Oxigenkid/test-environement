import time

import numpy as np
import pandas as pd
import pytest

from trader.agents.base import Context
from trader.agents.execution import ExecutionAgent
from trader.agents.learner import LearnerAgent
from trader.agents.risk import RiskAgent
from trader.broker import PaperBroker
from trader.config import load_config
from trader.journal import Journal
from trader.learning import (ADAPTIVE_KEY, active_cooldowns, candidate_params, current_params, diagnose,
                             effective_config, losing_streak, split_data, tune)
from trader.strategies import Signal


@pytest.fixture
def cfg():
    return load_config(None, {"db_path": ":memory:"})


@pytest.fixture
def journal():
    return Journal(":memory:")


def sig(**kw):
    base = dict(symbol="BTC/USD", strategy="trend", regime="trend_up", price=100.0, stop=97.0, target=110.0,
                strength=1.0, trail_atr=3.0)
    base.update(kw)
    return Signal(**base)


def synthetic(n=900, seed=1, drift=0.0006):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    high = close * (1 + rng.uniform(0, 0.006, n))
    low = close * (1 - rng.uniform(0, 0.006, n))
    open_ = np.r_[close[0], close[:-1]]
    idx = np.arange(n) * 14_400_000 + 1_700_000_000_000
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def add_closed(journal, symbol, strategy, pnl, i, highest=None, reason="stop"):
    journal.execute(
        "INSERT INTO trades(mode,symbol,strategy,regime,entry_ts,entry_price,qty,stop,target,trail_atr,highest,"
        "entry_fee,exit_ts,exit_price,exit_fee,pnl,exit_reason) VALUES('paper',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (symbol, strategy, "trend_up", 1000 + i, 100.0, 1.0, 97.0, 110.0, 3.0, highest or 100.0, 0.4,
         2000 + i, 100.0 + pnl, 0.4, pnl, reason))


def test_full_allocation_execution_invests_everything(cfg, journal):
    broker = PaperBroker(cfg, journal)
    ex = ExecutionAgent(cfg, journal, broker)
    ctx = Context(now=time.time(), prices={"BTC/USD": 100.0}, signals=[sig()])
    ex.run(ctx)
    assert len(journal.open_trades("paper")) == 1
    equity, cash = broker.equity({"BTC/USD": 100.0})
    assert cash < 0.01 * cfg.paper_starting_balance  # >99% committed


def test_full_allocation_splits_across_simultaneous_signals(cfg, journal):
    broker = PaperBroker(cfg, journal)
    ex = ExecutionAgent(cfg, journal, broker)
    prices = {"BTC/USD": 100.0, "ETH/USD": 50.0}
    ctx = Context(now=time.time(), prices=prices,
                  signals=[sig(), sig(symbol="ETH/USD", price=50.0, stop=48.0, target=56.0)])
    ex.run(ctx)
    trades = journal.open_trades("paper")
    assert len(trades) == 2
    costs = [t["qty"] * t["entry_price"] for t in trades]
    assert costs[0] == pytest.approx(costs[1], rel=0.02)
    assert broker.cash() < 0.01 * cfg.paper_starting_balance


def test_diagnose_kinds():
    base = {"entry_price": 100.0, "qty": 1.0, "target": 110.0, "entry_fee": 0.4, "exit_fee": 0.4}
    assert diagnose({**base, "pnl": -3.0, "highest": 100.1, "exit_reason": "stop"})["kind"] == "wrong_from_start"
    assert diagnose({**base, "pnl": -2.0, "highest": 106.0, "exit_reason": "stop"})["kind"] == "gave_back_gains"
    assert diagnose({**base, "pnl": -0.5, "highest": 101.0, "exit_reason": "time_stop"})["kind"] == "fee_drag"
    assert diagnose({**base, "pnl": -3.0, "highest": 102.0, "exit_reason": "stop"})["kind"] == "stopped_out"


def test_losing_streak():
    closed = [{"symbol": "BTC/USD", "strategy": "trend", "pnl": -1}, {"symbol": "ETH/USD", "strategy": "trend", "pnl": -1},
              {"symbol": "BTC/USD", "strategy": "trend", "pnl": -1}, {"symbol": "BTC/USD", "strategy": "trend", "pnl": 2}]
    assert losing_streak(closed) == 3
    assert losing_streak(closed, "BTC/USD", "trend") == 2
    assert losing_streak(closed, "ETH/USD", "trend") == 1


def test_effective_config_applies_learned_params(cfg, journal):
    assert effective_config(cfg, journal) is cfg
    journal.set_state(ADAPTIVE_KEY, {"stop_atr_mult": 4.0, "target_atr_mult": 9.0, "max_open_positions": 1,
                                     "strategies": ["trend"]})
    eff = effective_config(cfg, journal)
    assert eff.risk["stop_atr_mult"] == 4.0 and eff.risk["max_open_positions"] == 1
    assert eff.strategies == ["trend"]
    assert eff.risk["fee_rate"] == cfg.risk["fee_rate"]  # untouched values survive
    assert current_params(eff)["target_atr_mult"] == 9.0


def test_candidates_require_reward_above_risk(cfg):
    for p in candidate_params(cfg.learning["grid"]):
        assert p["target_atr_mult"] > p["stop_atr_mult"]


def test_split_data_keeps_warmup():
    data = {"BTC/USD": synthetic(900)}
    train, val = split_data(data)
    assert len(train["BTC/USD"]) == 600
    assert val["BTC/USD"].index[0] < train["BTC/USD"].index[-1]  # validation gets indicator warm-up
    assert val["BTC/USD"].index[-1] == data["BTC/USD"].index[-1]


def test_tune_returns_a_decision(cfg):
    c = load_config(None, {"db_path": ":memory:", "htf_sma_days": 0, "learning": {"grid": {
        "stop_atr_mult": [2.0, 3.0], "target_atr_mult": [6.0], "strategies": [["trend"]], "max_open_positions": [1]}}})
    data = {"BTC/USD": synthetic(900, 1), "ETH/USD": synthetic(900, 2)}
    res = tune(c, data)
    assert res["candidates"] == 2
    assert isinstance(res["adopted"], bool)
    assert res["best"]["score"] >= res["top"][-1]["score"]


def test_learner_post_mortem_cooldown_and_retune(journal):
    cfg = load_config(None, {"db_path": ":memory:", "htf_sma_days": 0, "learning": {
        "max_consecutive_losses": 2, "retune_after_losses": 2, "min_improvement": -10.0,  # always adopt
        "grid": {"stop_atr_mult": [2.0], "target_atr_mult": [9.0], "strategies": [["trend"]],
                 "max_open_positions": [1]}}})
    data = {"BTC/USD": synthetic(900, 1), "ETH/USD": synthetic(900, 2)}
    learner = LearnerAgent(cfg, journal, "paper", loader=lambda c: (data, None), background=False)
    now = time.time()
    learner.run(Context(now=now))  # initialises the trade cursor
    add_closed(journal, "BTC/USD", "trend", -3.0, 1)
    add_closed(journal, "BTC/USD", "trend", -2.5, 2, highest=106.0)
    note = learner.run(Context(now=now))
    assert len(journal.query("SELECT * FROM lessons")) == 2
    assert ("BTC/USD", "trend") in active_cooldowns(journal, now)
    assert "re-tuning" in note
    assert journal.get_state(ADAPTIVE_KEY)["target_atr_mult"] == 9.0
    assert len(journal.query("SELECT * FROM tunings")) == 1
    # the strategist / execution now refuse the benched pair
    broker = PaperBroker(cfg, journal)
    ExecutionAgent(cfg, journal, broker).run(Context(now=now, prices={"BTC/USD": 100.0}, signals=[sig()]))
    assert not journal.open_trades("paper")


def test_kill_switch_retunes_then_auto_resumes(journal):
    cfg = load_config(None, {"db_path": ":memory:", "htf_sma_days": 0, "learning": {
        "resume_after_hours": 1, "grid": {"stop_atr_mult": [3.0], "target_atr_mult": [6.0],
                                          "strategies": [["trend"]], "max_open_positions": [1]}}})
    data = {"BTC/USD": synthetic(900, 1)}
    broker = PaperBroker(cfg, journal)
    ex = ExecutionAgent(cfg, journal, broker)
    risk = RiskAgent(cfg, journal, broker)
    learner = LearnerAgent(cfg, journal, "paper", loader=lambda c: (data, None), background=False)
    now = time.time()
    learner.run(Context(now=now))
    ex.run(Context(now=now, prices={"BTC/USD": 100.0}, signals=[sig()]))
    risk.run(Context(now=now, prices={"BTC/USD": 100.0}))
    crash = Context(now=now, prices={"BTC/USD": 60.0})
    risk.run(crash)
    ex.run(crash)
    assert journal.get_state("halted") and not journal.open_trades("paper")
    learner.run(crash)
    assert journal.get_state("halt_resume_at") == pytest.approx(now + 3600)
    assert len(journal.query("SELECT * FROM tunings WHERE reason LIKE 'kill-switch%'")) == 1
    early = Context(now=now + 600, prices={"BTC/USD": 60.0})
    risk.run(early)
    assert journal.get_state("halted") and not early.allow_entries
    later = Context(now=now + 3700, prices={"BTC/USD": 60.0})
    risk.run(later)
    assert not journal.get_state("halted") and later.allow_entries
    assert journal.get_state("paper_peak") == pytest.approx(broker.equity({})[0])


def test_live_does_not_auto_resume_by_default(cfg, journal):
    class LiveStub(PaperBroker):
        mode = "live"
    broker = LiveStub(cfg, journal)
    journal.set_state("halted", "kill-switch: test")
    journal.set_state("halt_resume_at", time.time() - 10)
    ctx = Context(now=time.time(), prices={})
    RiskAgent(cfg, journal, broker).run(ctx)
    assert journal.get_state("halted") and not ctx.allow_entries


def test_commits_when_flat_and_rotates_after_loss(cfg, journal):
    from trader.agents.strategist import StrategistAgent
    from trader.indicators import add_indicators

    df = add_indicators(synthetic(300, 1, drift=-0.0002))
    df["htf_ok"] = 0.0  # bear regime that previously blocked 100% of trades
    bar_ts = str(int(df.index[-1]))
    journal.set_state("last_signal_bar", {"BTC/USD": bar_ts})  # bar already evaluated earlier
    px = float(df["close"].iloc[-1])

    broker = PaperBroker(cfg, journal)
    strat = StrategistAgent(cfg, journal, "paper")
    ex = ExecutionAgent(cfg, journal, broker)
    learner = LearnerAgent(cfg, journal, "paper", background=False, loader=lambda c: ({"BTC/USD": df}, None))
    now = time.time()
    learner.run(Context(now=now))

    # Beat 1: even in bear regime on an already-evaluated bar, commits the whole amount immediately
    c1 = Context(now=now, candles={"BTC/USD": df}, prices={"BTC/USD": px})
    strat.run(c1)
    assert len(c1.signals) == 1
    first_strat = c1.signals[0].strategy
    ex.run(c1)
    open_t = journal.open_trades("paper")
    assert len(open_t) == 1 and broker.cash() < 0.01 * cfg.paper_starting_balance

    # Beat 2: price drops below stop -> exits at a loss -> learner benches first_strat & adapts -> rotates strategy
    stop_px = open_t[0]["stop"] * 0.99
    c2 = Context(now=now + 60, candles={"BTC/USD": df}, prices={"BTC/USD": stop_px})
    ex.ctx_now = c2.now
    ex.manage_exits(c2)
    c2.exits_done = True
    learner.run(c2)
    assert ("BTC/USD", first_strat) in active_cooldowns(journal, c2.now)
    assert journal.get_state(ADAPTIVE_KEY) is not None
    strat.run(c2)
    ex.run(c2)
    open_t2 = journal.open_trades("paper")
    assert len(open_t2) == 1 and open_t2[0]["strategy"] != first_strat


def test_sol_excluded_and_existing_sol_trade_closed(journal):
    cfg = load_config(None, {"db_path": ":memory:", "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"]})
    assert "SOL/USD" not in cfg.symbols
    assert cfg.symbols == ["BTC/USD", "ETH/USD"]

    broker = PaperBroker(cfg, journal)
    ex = ExecutionAgent(cfg, journal, broker)
    journal.execute(
        "INSERT INTO trades(mode,symbol,strategy,regime,entry_ts,entry_price,qty,stop,target) "
        "VALUES('paper','SOL/USD','trend','trend_up',1000,150.0,0.05,140.0,170.0)"
    )
    assert len(journal.open_trades("paper")) == 1
    ex.run(Context(now=time.time(), prices={"SOL/USD": 150.0, "BTC/USD": 100.0}))
    assert not journal.open_trades("paper")
    assert journal.closed_trades("paper")[0]["exit_reason"] == "symbol_removed"


