from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _last_valid(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _ret_n(close: pd.Series, n: int) -> float | None:
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) <= n:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0)


def _ma(close: pd.Series, n: int) -> pd.Series:
    s = pd.to_numeric(close, errors="coerce")
    return s.rolling(n).mean()


@dataclass(frozen=True)
class MarketEnv:
    allow_attack: bool
    hs300_breakdown: bool
    gem_crash: bool
    sse_downtrend: bool
    systemic_risk: bool
    detail: dict


def compute_market_env(index_map: dict[str, pd.DataFrame]) -> MarketEnv:
    detail: dict = {}
    hs300 = index_map.get("沪深300")
    sse = index_map.get("上证指数")
    gem = index_map.get("创业板指")
    kcb = index_map.get("科创50")

    def idx_flags(df: pd.DataFrame | None) -> dict:
        if df is None or df.empty:
            return {"ok": False}
        close = pd.to_numeric(df["close"], errors="coerce")
        ma20 = _ma(close, 20)
        last_close = close.dropna().iloc[-1] if close.dropna().shape[0] else np.nan
        last_ma20 = ma20.dropna().iloc[-1] if ma20.dropna().shape[0] else np.nan
        ret5 = _ret_n(close, 5)
        below_ma20 = bool(pd.notna(last_close) and pd.notna(last_ma20) and last_close < last_ma20)
        return {
            "ok": True,
            "ret5": ret5,
            "below_ma20": below_ma20,
        }

    hs = idx_flags(hs300)
    ss = idx_flags(sse)
    gm = idx_flags(gem)
    kc = idx_flags(kcb)
    detail["沪深300"] = hs
    detail["上证指数"] = ss
    detail["创业板指"] = gm
    detail["科创50"] = kc

    hs300_breakdown = bool(hs.get("ok") and hs.get("below_ma20") and (hs.get("ret5") is not None and hs["ret5"] < -0.02))
    gem_crash = bool(gm.get("ok") and (gm.get("ret5") is not None and gm["ret5"] < -0.03))
    sse_downtrend = bool(ss.get("ok") and ss.get("below_ma20"))
    systemic_risk = hs300_breakdown or (gem_crash and sse_downtrend)

    allow_attack = not systemic_risk and not hs.get("below_ma20", False)
    return MarketEnv(
        allow_attack=allow_attack,
        hs300_breakdown=hs300_breakdown,
        gem_crash=gem_crash,
        sse_downtrend=sse_downtrend,
        systemic_risk=systemic_risk,
        detail=detail,
    )


def compute_indicators(etf_df: pd.DataFrame, bench_df: pd.DataFrame | None) -> dict:
    out: dict = {}
    if etf_df is None or etf_df.empty:
        return out

    close = pd.to_numeric(etf_df["close"], errors="coerce")
    amount = pd.to_numeric(etf_df.get("amount"), errors="coerce") if "amount" in etf_df.columns else pd.Series(dtype=float)
    pct = pd.to_numeric(etf_df.get("pct_chg"), errors="coerce") if "pct_chg" in etf_df.columns else pd.Series(dtype=float)

    out["etf_ret_5"] = _ret_n(close, 5)
    out["etf_ret_10"] = _ret_n(close, 10)
    out["etf_ret_20"] = _ret_n(close, 20)

    ma20 = _ma(close, 20)
    out["ma20"] = float(ma20.dropna().iloc[-1]) if ma20.dropna().shape[0] else None
    out["ma20_prev"] = float(ma20.dropna().iloc[-2]) if ma20.dropna().shape[0] >= 2 else None
    out["close"] = _last_valid(etf_df, "close")
    out["amount"] = _last_valid(etf_df, "amount")
    out["pct_chg"] = _last_valid(etf_df, "pct_chg")

    if out.get("close") is not None and out.get("ma20") is not None:
        out["dist_ma20"] = float(out["close"] / out["ma20"] - 1.0)
        out["above_ma20"] = out["close"] >= out["ma20"]
        out["ma20_up"] = (out.get("ma20_prev") is not None) and (out["ma20"] >= out["ma20_prev"])
    else:
        out["dist_ma20"] = None
        out["above_ma20"] = None
        out["ma20_up"] = None

    if pct.dropna().shape[0] >= 3:
        last3 = pct.dropna().iloc[-3:]
        big_up = (last3 > 0.015).all()
        big_down = (last3 < -0.015).all()
    else:
        big_up = False
        big_down = False
    out["consecutive_big_up_3"] = bool(big_up)
    out["consecutive_down_3"] = bool(big_down)

    if amount.dropna().shape[0] >= 6:
        last = float(amount.dropna().iloc[-1])
        avg5 = float(amount.dropna().iloc[-6:-1].mean())
        out["amount_avg5"] = avg5
        out["amount_up"] = (avg5 > 0) and (last > avg5 * 1.2)
    else:
        out["amount_avg5"] = None
        out["amount_up"] = None

    if amount.dropna().shape[0] >= 2 and pct.dropna().shape[0] >= 2:
        out["volume_down"] = bool(amount.dropna().iloc[-1] > amount.dropna().iloc[-2] and pct.dropna().iloc[-1] < 0)
    else:
        out["volume_down"] = None

    if out.get("amount_up") and out.get("pct_chg") is not None:
        out["high_volume_stagnation"] = abs(out["pct_chg"]) <= 0.003
    else:
        out["high_volume_stagnation"] = False

    out["support_ma20_ok"] = None
    if close.dropna().shape[0] >= 25 and ma20.dropna().shape[0] >= 1:
        last_ma20 = float(ma20.dropna().iloc[-1])
        last5_close = close.dropna().iloc[-5:]
        out["support_ma20_ok"] = bool((last5_close >= last_ma20 * 0.99).all())

    if bench_df is not None and (bench_df is not None and not bench_df.empty) and "close" in bench_df.columns:
        bclose = pd.to_numeric(bench_df["close"], errors="coerce")
        out["bench_ret_5"] = _ret_n(bclose, 5)
        out["bench_ret_10"] = _ret_n(bclose, 10)
        out["bench_ret_20"] = _ret_n(bclose, 20)
        out["bench_pct_chg"] = _last_valid(bench_df, "pct_chg")
    else:
        out["bench_ret_5"] = None
        out["bench_ret_10"] = None
        out["bench_ret_20"] = None
        out["bench_pct_chg"] = None

    def outperf(n: int) -> bool | None:
        er = out.get(f"etf_ret_{n}")
        br = out.get(f"bench_ret_{n}")
        if er is None or br is None:
            return None
        return er > br

    out["outperf_5"] = outperf(5)
    out["outperf_10"] = outperf(10)
    out["outperf_20"] = outperf(20)
    return out


def score_buy(ind: dict, env: MarketEnv) -> tuple[float, dict]:
    breakdown: dict = {"趋势分": {}, "位置分": {}, "资金分": {}, "大盘环境分": {}, "情绪风险分": {}}
    total = 0.0

    def add(group: str, key: str, pts: float, ok: bool | None, note_ok: str, note_no: str):
        nonlocal total
        if ok is None:
            breakdown[group][key] = {"分数": 0, "触发": "未知", "说明": "数据不足"}
            return
        if ok:
            total += pts
            breakdown[group][key] = {"分数": pts, "触发": "是", "说明": note_ok}
        else:
            breakdown[group][key] = {"分数": 0, "触发": "否", "说明": note_no}

    add("趋势分", "近5日跑赢沪深300", 8, ind.get("outperf_5"), "跑赢基准", "未跑赢基准")
    add("趋势分", "近10日跑赢沪深300", 8, ind.get("outperf_10"), "跑赢基准", "未跑赢基准")
    add("趋势分", "近20日跑赢沪深300", 6, ind.get("outperf_20"), "跑赢基准", "未跑赢基准")
    add("趋势分", "站上20日均线", 4, ind.get("above_ma20"), "价格在20日均线之上", "价格在20日均线之下")
    add("趋势分", "20日均线向上", 4, ind.get("ma20_up"), "均线向上", "均线走平或向下")

    dist = ind.get("dist_ma20")
    pts = 0
    label = "距离20日均线"
    if dist is None:
        breakdown["位置分"][label] = {"分数": 0, "触发": "未知", "说明": "数据不足"}
    else:
        if 0 <= dist <= 0.03:
            pts = 12
            note = "离均线不远，位置舒适"
        elif 0.03 < dist <= 0.06:
            pts = 8
            note = "略有拉升，仍可接受"
        elif 0.06 < dist <= 0.10:
            pts = 2
            note = "有一定追高风险"
        else:
            pts = -5
            note = "明显偏离均线，追高风险大"
        total += pts
        breakdown["位置分"][label] = {"分数": pts, "触发": f"{dist:.2%}", "说明": note}

    add(
        "位置分",
        "近3日没有连续大涨",
        5,
        not ind.get("consecutive_big_up_3", False),
        "没有连续大涨",
        "连续大涨，容易分歧",
    )
    add(
        "位置分",
        "回踩未破关键均线",
        5,
        ind.get("support_ma20_ok"),
        "回踩后仍守住均线附近",
        "回踩力度较大",
    )

    add("资金分", "成交额较5日均量放大", 6, ind.get("amount_up"), "资金放大", "资金未放大")
    stronger = None
    if ind.get("pct_chg") is not None and ind.get("bench_pct_chg") is not None:
        stronger = ind["pct_chg"] > ind["bench_pct_chg"]
    add("资金分", "板块涨幅强于沪深300", 5, stronger, "强于大盘", "弱于大盘")
    day_up = ind.get("pct_chg") is not None and ind["pct_chg"] > 0
    add("资金分", "当日上涨且成交额放大", 5, bool(day_up and ind.get("amount_up")), "价量齐升", "未形成价量齐升")
    add(
        "资金分",
        "没有明显放量滞涨",
        4,
        not ind.get("high_volume_stagnation", False),
        "没有放量滞涨",
        "出现放量滞涨迹象",
    )

    add("大盘环境分", "沪深300没有破位", 4, not env.hs300_breakdown, "大盘未明显破位", "大盘出现破位")
    add("大盘环境分", "创业板/科创板没有大跌", 4, not env.gem_crash, "成长未大跌", "成长出现大跌")
    add("大盘环境分", "上证指数没有系统性下跌", 3, not env.sse_downtrend, "上证未明显走弱", "上证走弱")
    add("大盘环境分", "市场没有明显系统性风险", 4, not env.systemic_risk, "系统性风险低", "系统性风险偏高")

    started_or_repair = None
    if ind.get("etf_ret_5") is not None and ind.get("etf_ret_20") is not None:
        started_or_repair = (ind["etf_ret_5"] > 0) and (ind["etf_ret_20"] < 0.15)
    add("情绪风险分", "刚启动或修复", 5, started_or_repair, "情绪偏修复/启动", "可能已进入高位阶段")
    add("情绪风险分", "没有连续暴涨三天", 3, not ind.get("consecutive_big_up_3", False), "未连续暴涨", "连续暴涨三天")
    add(
        "情绪风险分",
        "没有高位放量滞涨",
        2,
        not (ind.get("high_volume_stagnation", False) and (dist is not None and dist > 0.08)),
        "未出现高位放量滞涨",
        "出现高位放量滞涨",
    )

    if ind.get("consecutive_big_up_3", False):
        total -= 5
        breakdown["情绪风险分"]["连续暴涨三天扣分"] = {"分数": -5, "触发": "是", "说明": "连续暴涨三天，风险上升"}
    if ind.get("high_volume_stagnation", False) and (dist is not None and dist > 0.08):
        total -= 5
        breakdown["情绪风险分"]["高位放量滞涨扣分"] = {"分数": -5, "触发": "是", "说明": "高位放量滞涨，需谨慎"}

    total = float(np.clip(total, 0, 100))
    return total, breakdown


def score_sell(
    ind: dict,
    env: MarketEnv,
    *,
    holding_position_ratio: float | None = None,
    holding_pnl: float | None = None,
) -> tuple[float, dict]:
    breakdown: dict = {
        "趋势破坏分": {},
        "资金流出分": {},
        "高位风险分": {},
        "大盘风险分": {},
        "持仓风险分": {},
        "替代机会分": {},
    }
    total = 0.0

    def add(group: str, key: str, pts: float, ok: bool | None, note_ok: str, note_no: str):
        nonlocal total
        if ok is None:
            breakdown[group][key] = {"分数": 0, "触发": "未知", "说明": "数据不足"}
            return
        if ok:
            total += pts
            breakdown[group][key] = {"分数": pts, "触发": "是", "说明": note_ok}
        else:
            breakdown[group][key] = {"分数": 0, "触发": "否", "说明": note_no}

    below_ma20 = ind.get("above_ma20")
    below_ma20 = None if below_ma20 is None else (not below_ma20)
    add("趋势破坏分", "跌破20日均线", 10, below_ma20, "跌破20日均线", "未跌破20日均线")

    two_days_no_recover = None
    if ind.get("above_ma20") is not None and ind.get("ma20") is not None:
        two_days_no_recover = bool(below_ma20)
    add("趋势破坏分", "跌破后2天未收回(近似)", 10, two_days_no_recover, "破位后未收回", "未满足")

    underperf5 = None
    if ind.get("outperf_5") is not None:
        underperf5 = not ind["outperf_5"]
    underperf10 = None
    if ind.get("outperf_10") is not None:
        underperf10 = not ind["outperf_10"]
    add("趋势破坏分", "近5日跑输沪深300", 5, underperf5, "跑输基准", "未跑输基准")
    add("趋势破坏分", "近10日跑输沪深300", 5, underperf10, "跑输基准", "未跑输基准")

    add("资金流出分", "放量下跌", 8, ind.get("volume_down"), "放量下跌", "未出现放量下跌")
    add(
        "资金流出分",
        "放量但价格不涨",
        5,
        ind.get("high_volume_stagnation"),
        "成交放大但涨幅不明显",
        "未出现",
    )
    weaker = None
    if ind.get("pct_chg") is not None and ind.get("bench_pct_chg") is not None:
        weaker = ind["pct_chg"] < ind["bench_pct_chg"] - 0.005
    add("资金流出分", "板块整体弱于大盘", 4, weaker, "弱于大盘", "未明显弱于大盘")
    add(
        "资金流出分",
        "龙头走弱(近似)",
        3,
        bool(below_ma20 and (ind.get("pct_chg") is not None and ind["pct_chg"] < 0)),
        "弱势结构未改",
        "未满足",
    )

    dist = ind.get("dist_ma20")
    pullback = None
    if dist is not None and ind.get("close") is not None:
        pullback = dist > 0.10 and (ind.get("pct_chg") is not None and ind["pct_chg"] < 0)
    add("高位风险分", "远离均线后回落", 5, pullback, "高位回落", "未满足")
    add(
        "高位风险分",
        "近5日涨幅过大",
        5,
        ind.get("etf_ret_5") is not None and ind["etf_ret_5"] > 0.10,
        "短期涨幅过大",
        "未满足",
    )
    add(
        "高位风险分",
        "高位放量滞涨",
        5,
        bool(ind.get("high_volume_stagnation") and (dist is not None and dist > 0.08)),
        "高位放量滞涨",
        "未满足",
    )

    add("大盘风险分", "沪深300破位", 5, env.hs300_breakdown, "大盘破位", "未破位")
    add("大盘风险分", "创业板/科创板大跌", 5, env.gem_crash, "成长大跌", "未大跌")
    add("大盘风险分", "市场普跌(系统性风险)", 5, env.systemic_risk, "系统性风险偏高", "系统性风险不明显")

    if holding_pnl is None:
        breakdown["持仓风险分"]["浮盈回撤/亏损(简化)"] = {"分数": 0, "触发": "未知", "说明": "未填写盈亏"}
    else:
        if holding_pnl <= -0.05:
            total += 5
            breakdown["持仓风险分"]["浮盈回撤/亏损(简化)"] = {"分数": 5, "触发": f"{holding_pnl:.2%}", "说明": "亏损扩大，需控制风险"}
        else:
            breakdown["持仓风险分"]["浮盈回撤/亏损(简化)"] = {"分数": 0, "触发": f"{holding_pnl:.2%}", "说明": "未触发"}

    if holding_position_ratio is None:
        breakdown["持仓风险分"]["仓位过重"] = {"分数": 0, "触发": "未知", "说明": "无法计算仓位"}
    else:
        if holding_position_ratio >= 0.22:
            total += 5
            breakdown["持仓风险分"]["仓位过重"] = {"分数": 5, "触发": f"{holding_position_ratio:.2%}", "说明": "仓位偏重，风险放大"}
        else:
            breakdown["持仓风险分"]["仓位过重"] = {"分数": 0, "触发": f"{holding_position_ratio:.2%}", "说明": "未触发"}

    breakdown["替代机会分"]["其他赛道更强(由决策层处理)"] = {"分数": 0, "触发": "未知", "说明": "由最终动作决策层评估"}

    total = float(np.clip(total, 0, 100))
    return total, breakdown

