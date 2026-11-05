"""Local dashboard (FastAPI). Read-only view of the journal; binds to 127.0.0.1 by default."""
from __future__ import annotations

import json
import math
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import Config
from .gate import check_gate
from .journal import Journal
from .learning import effective_config


def _clean(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="Autonomous trader")
    j = Journal(cfg.db_path)

    def mode() -> str:
        return (j.get_state("heartbeat") or {}).get("mode") or cfg.mode

    @app.get("/api/status")
    def status():
        hb = j.get_state("heartbeat") or {}
        m = mode()
        ok, stats, problems = check_gate(cfg, j)
        peak = j.get_state(f"{m}_peak")
        return _clean({
            "now": time.time(), "heartbeat": hb, "mode": m, "exchange": cfg.exchange, "symbols": cfg.symbols,
            "timeframe": cfg.timeframe, "target": cfg.target_equity, "start": cfg.paper_starting_balance,
            "peak": peak, "halted": j.get_state("halted"), "goal": j.get_state("goal_reached"),
            "regimes": j.get_state("regimes") or {}, "intel": j.get_state("intel") or {},
            "mood": j.get_state("mood") or {}, "weights": j.get_state("strategy_weights") or {},
            "agents": j.get_state("agent_health") or {}, "risk": dict(effective_config(cfg, j).risk),
            "gate": {"passed": ok, "stats": stats, "problems": problems},
            "learning": {
                "params": j.get_state("adaptive_params"), "cooldowns": j.get_state("cooldowns") or {},
                "last_tuning": j.get_state("last_tuning_result"), "auto_resume_at": j.get_state("halt_resume_at"),
                "lessons": j.query("SELECT ts, symbol, strategy, kind, lesson, pnl FROM lessons ORDER BY ts DESC LIMIT 10"),
            },
        })

    @app.get("/api/trades")
    def trades():
        m = mode()
        return _clean({"open": j.open_trades(m), "closed": j.closed_trades(m, limit=50)})

    @app.get("/api/equity")
    def equity():
        rows = j.query("SELECT ts, equity FROM equity WHERE mode=? ORDER BY ts", (mode(),))
        step = max(1, len(rows) // 400)
        return _clean(rows[::step] + (rows[-1:] if rows and len(rows) % step else []))

    @app.get("/api/news")
    def news():
        rows = j.query("SELECT title, link, source, published, score, symbols FROM news ORDER BY published DESC LIMIT 40")
        for r in rows:
            r["symbols"] = json.loads(r["symbols"])
        return _clean(rows)

    @app.get("/api/events")
    def events():
        return _clean(j.query("SELECT * FROM events ORDER BY ts DESC LIMIT 60"))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    return app


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Autonomous trader</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--text:#e6e8ee;--muted:#8b93a7;--up:#2fbf71;--down:#e5534b;--warn:#d4a72c;--accent:#6ea8fe}
@media (prefers-color-scheme: light){:root{--bg:#f6f7f9;--panel:#fff;--line:#e3e6ec;--text:#161a22;--muted:#5d6577;--up:#1a7f4b;--down:#c4312a;--warn:#9a6b00;--accent:#2f6fe0}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0}h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 10px}
main{padding:16px 20px;display:grid;gap:16px;grid-template-columns:repeat(12,1fr)}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;grid-column:span 12;min-width:0}
@media(min-width:1000px){.s4{grid-column:span 4}.s6{grid-column:span 6}.s8{grid-column:span 8}}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.m .l{color:var(--muted);font-size:12px}.m .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--line);font-size:12px}
.ok{color:var(--up)}.bad{color:var(--down)}.warn{color:var(--warn)}.muted{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:12px}.wrap{white-space:normal}.scroll{overflow:auto;max-height:360px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>div{height:100%;background:var(--accent)}
svg{width:100%;height:220px;display:block}a{color:var(--accent);text-decoration:none}
</style></head><body>
<header><h1>Autonomous trader</h1><div id="hb" class="muted">connecting...</div></header>
<main>
<section><div class="metrics" id="metrics"></div>
<div style="margin-top:12px"><div class="muted" id="progress-label" style="font-size:12px;margin-bottom:4px"></div><div class="bar"><div id="progress" style="width:0"></div></div></div>
<div id="alerts" style="margin-top:10px"></div></section>
<section class="s8"><h2>Equity</h2><svg id="chart" role="img" aria-label="Equity curve"></svg></section>
<section class="s4"><h2>Agents</h2><div class="scroll"><table id="agents"></table></div></section>
<section class="s6"><h2>Market intelligence</h2><table id="intel"></table><p class="muted" id="mood"></p></section>
<section class="s6"><h2>Go-live gate</h2><div id="gate"></div></section>
<section class="s12"><h2>Self-improvement &amp; lessons</h2><p class="muted" id="learn-summary"></p><div class="scroll"><table id="lessons"></table></div></section>
<section class="s6"><h2>Open positions</h2><div class="scroll"><table id="open"></table></div></section>
<section class="s6"><h2>Closed trades</h2><div class="scroll"><table id="closed"></table></div></section>
<section class="s6"><h2>Latest headlines</h2><div class="scroll"><table id="news"></table></div></section>
<section class="s6"><h2>Event log</h2><div class="scroll"><table id="events"></table></div></section>
</main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const money=v=>v==null?'-':'$'+Number(v).toFixed(Number(v)<100?4:2);
const t=ts=>ts?new Date(ts*1000).toLocaleString():'-';
const ago=ts=>{const s=Math.round(Date.now()/1000-ts);return s<90?s+'s ago':s<5400?Math.round(s/60)+'m ago':Math.round(s/3600)+'h ago'};
const sc=v=>`<span class="${v>0.05?'ok':v<-0.05?'bad':'muted'}">${v>=0?'+':''}${Number(v).toFixed(2)}</span>`;
function table(el,head,rows){$(el).innerHTML='<tr>'+head.map(h=>`<th>${h}</th>`).join('')+'</tr>'+(rows.length?rows.join(''):`<tr><td class="muted" colspan="${head.length}">none yet</td></tr>`)}
function chart(pts){const svg=$('chart');if(pts.length<2){svg.innerHTML='<text x="10" y="20" fill="currentColor" opacity=".6">Waiting for data...</text>';return}
const W=svg.clientWidth||600,H=220,p=30,xs=pts.map(d=>d.ts),ys=pts.map(d=>d.equity),x0=Math.min(...xs),x1=Math.max(...xs);let y0=Math.min(...ys),y1=Math.max(...ys);if(y1-y0<1e-9){y0-=.5;y1+=.5}
const X=v=>p+(v-x0)/(x1-x0||1)*(W-p-8),Y=v=>H-20-(v-y0)/(y1-y0)*(H-40);
const d=pts.map((q,i)=>(i?'L':'M')+X(q.ts).toFixed(1)+','+Y(q.equity).toFixed(1)).join('');
svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
svg.innerHTML=`<text x="2" y="${Y(y1)+4}" font-size="11" fill="currentColor" opacity=".6">${money(y1)}</text><text x="2" y="${Y(y0)+4}" font-size="11" fill="currentColor" opacity=".6">${money(y0)}</text><path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2"/>`}
async function refresh(){try{
const [s,tr,eq,nw,ev]=await Promise.all(['status','trades','equity','news','events'].map(u=>fetch('/api/'+u).then(r=>r.json())));
const hb=s.heartbeat||{};const age=hb.ts?Date.now()/1000-hb.ts:1e9;const alive=age<(hb.interval||60)*3;
$('hb').innerHTML=`<span class="pill ${alive?'ok':'bad'}">${alive?'● alive':'● no heartbeat'}</span> ${esc(s.mode).toUpperCase()} · ${esc(s.exchange)} · ${esc(s.timeframe)} · ${esc(s.risk.allocation||'full')} alloc · beat #${hb.beats||0} ${hb.ts?ago(hb.ts):''}`;
const eqv=hb.equity||0,start=s.start,dd=s.peak?1-eqv/s.peak:0;
$('metrics').innerHTML=[['Equity',money(eqv)],['Return vs start',start?((eqv/start-1)*100).toFixed(1)+'%':'-'],['Peak',money(s.peak)],['Drawdown',(dd*100).toFixed(1)+'% / '+(s.risk.kill_switch_drawdown*100)+'%'],['Target',money(s.target)],['Open positions',tr.open.length+' / '+s.risk.max_open_positions],['Allocation',esc(s.risk.allocation||'full')]].map(([l,v])=>`<div class="m"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('');
const prog=Math.max(0,Math.min(1,Math.log(Math.max(eqv,start)/start)/Math.log(s.target/start)));$('progress').style.width=(prog*100).toFixed(1)+'%';
$('progress-label').textContent=`Capital ladder progress to ${money(s.target)} (log scale): ${(prog*100).toFixed(1)}%`;
const lr=s.learning||{};
$('alerts').innerHTML=(s.halted?`<p class="bad">HALTED: ${esc(s.halted)}${lr.auto_resume_at?' (auto-resume at '+t(lr.auto_resume_at)+')':''}</p>`:'')+(s.goal?`<p class="ok">Target reached at ${t(s.goal.ts)} with ${money(s.goal.equity)}</p>`:'');
table('agents',['Agent','Status','Last run','Note'],Object.entries(s.agents).map(([k,a])=>`<tr><td>${esc(k)}</td><td class="${a.ok?'ok':'bad'}">${a.ok?'ok':'error'}</td><td>${ago(a.last_run)}</td><td class="wrap muted">${esc(a.note).slice(0,140)}</td></tr>`));
table('intel',['Symbol','Price','Regime','Coin news','Market news','Size x','Veto'],s.symbols.map(sym=>{const i=s.intel[sym]||{};return `<tr><td>${esc(sym)}</td><td>${money((hb.prices||{})[sym])}</td><td>${esc(s.regimes[sym]||'-')}</td><td>${sc(i.coin||0)}</td><td>${sc(i.market||0)}</td><td>${i.mult!=null?i.mult.toFixed(2):'-'}</td><td class="${i.veto?'bad':'muted'} wrap">${esc(i.veto||'-')}</td></tr>`}));
const fg=(s.mood||{}).fear_greed,gl=(s.mood||{}).global;$('mood').textContent=(fg?`Fear & Greed ${fg.value} (${fg.label})`:'Fear & Greed n/a')+(gl?` · total market cap 24h ${gl.mcap_change_24h.toFixed(1)}% · BTC dominance ${gl.btc_dominance.toFixed(1)}%`:'')+(Object.keys(s.weights).length?' · strategy weights '+Object.entries(s.weights).map(([k,v])=>k+' '+v.toFixed(2)).join(', '):'');
const g=s.gate,st=g.stats;$('gate').innerHTML=`<p><span class="pill ${g.passed?'ok':'warn'}">${g.passed?'passed - live trading unlocked':'locked - keep paper trading'}</span></p><p class="muted">${st.trades} paper trades · win rate ${(st.win_rate*100).toFixed(0)}% · profit factor ${st.profit_factor===null||st.profit_factor===Infinity?'∞':Number(st.profit_factor).toFixed(2)} · max drawdown ${(st.max_drawdown*100).toFixed(1)}% · net ${money(st.net_pnl)}</p>`+g.problems.map(p=>`<div class="warn">• ${esc(p)}</div>`).join('');
const cds=Object.entries(lr.cooldowns||{}).filter(([,u])=>u>s.now).map(([k,u])=>`${k} until ${t(u)}`).join(', ');
const lp=lr.params?`stop ${lr.params.stop_atr_mult}ATR · target ${lr.params.target_atr_mult}ATR · max_pos ${lr.params.max_open_positions} · ${(lr.params.strategies||[]).join('+')}`:'config defaults';
$('learn-summary').textContent=`Active parameters: ${lp}`+(cds?` · Cool-downs: ${cds}`:' · Cool-downs: none')+(lr.last_tuning?` · Last re-tune (${lr.last_tuning.reason}): ${lr.last_tuning.adopted?'adopted':'kept current'}`:'');
table('lessons',['Time','Symbol','Strategy','P&L','Diagnosis','Lesson'],(lr.lessons||[]).map(l=>`<tr><td>${ago(l.ts)}</td><td>${esc(l.symbol)}</td><td>${esc(l.strategy)}</td><td class="bad">${money(l.pnl)}</td><td>${esc(l.kind)}</td><td class="wrap muted">${esc(l.lesson)}</td></tr>`));
table('open',['Symbol','Strategy','Entry','Now','Stop','Target','Qty','Opened'],tr.open.map(o=>{const px=(hb.prices||{})[o.symbol];return `<tr><td>${esc(o.symbol)}</td><td>${esc(o.strategy)}</td><td>${money(o.entry_price)}</td><td class="${px>=o.entry_price?'ok':'bad'}">${money(px)}</td><td>${money(o.stop)}</td><td>${o.trail_atr?'trailing':money(o.target)}</td><td>${Number(o.qty).toPrecision(4)}</td><td>${ago(o.entry_ts)}</td></tr>`}));
table('closed',['Symbol','Strategy','P&L','Exit reason','Closed'],tr.closed.map(c=>`<tr><td>${esc(c.symbol)}</td><td>${esc(c.strategy)}</td><td class="${c.pnl>=0?'ok':'bad'}">${money(c.pnl)}</td><td>${esc(c.exit_reason)}</td><td>${ago(c.exit_ts)}</td></tr>`));
table('news',['Score','Tags','Headline'],nw.map(n=>`<tr><td>${sc(n.score)}</td><td class="muted">${esc(n.symbols.join(','))}</td><td class="wrap"><a href="${esc(n.link)}" target="_blank" rel="noopener">${esc(n.title)}</a> <span class="muted">${esc(n.source)} · ${ago(n.published)}</span></td></tr>`));
table('events',['Time','Agent','Message'],ev.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${esc(e.agent)}</td><td class="wrap ${e.level==='error'?'bad':e.level==='warn'?'warn':''}">${esc(e.message)}</td></tr>`));
chart(eq);
}catch(e){$('hb').innerHTML='<span class="pill bad">dashboard error</span> '+esc(e)}}
refresh();setInterval(refresh,5000);
</script></body></html>"""
