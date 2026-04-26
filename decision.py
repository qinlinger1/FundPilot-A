from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from config import BUY_AMOUNT_RULES
from portfolio import PortfolioState, can_add_more, fund_position_ratio
from strategy import MarketEnv
from utils import TradingStatus, format_currency


@dataclass(frozen=True)
class FinalDecision:
    action: str
    target: str | None
    amount_or_ratio: str
    buy_score: float | None
    sell_score: float | None
    reason: dict
    forbidden: list[str]


def _pick_buy_amount(buy_score: float) -> int:
    for threshold, amount in BUY_AMOUNT_RULES:
        if buy_score >= threshold:
            return int(amount)
    return 0


def _top_reasons(breakdown_group: dict, n: int = 3) -> list[str]:
    items = []
    for k, v in (breakdown_group or {}).items():
        try:
            pts = float(v.get("分数", 0))
            if pts != 0:
                items.append((abs(pts), pts, k, v.get("说明")))
        except Exception:
            continue
    items.sort(reverse=True)
    out = []
    for _, pts, k, note in items[:n]:
        if note:
            out.append(f"{k}：{note}（{pts:+.0f}）")
        else:
            out.append(f"{k}（{pts:+.0f}）")
    return out


def decide(
    *,
    status: TradingStatus,
    env: MarketEnv,
    ranking_df: pd.DataFrame,
    portfolio: PortfolioState,
    now: datetime,
) -> FinalDecision:
    forbidden: list[str] = [
        "不要做自动下单",
        "不要承诺收益",
        "不要使用未来函数",
        "不要用盘后净值倒推盘中决策",
    ]

    if not status.is_trade_day:
        forbidden.append("非交易日不操作")
        return FinalDecision(
            action="不动",
            target=None,
            amount_or_ratio="-",
            buy_score=None,
            sell_score=None,
            reason={
                "趋势判断": ["非交易日"],
                "位置判断": [],
                "资金判断": [],
                "风险判断": ["非交易日，不生成买入建议"],
                "仓位判断": [],
            },
            forbidden=forbidden,
        )

    if status.is_after_close:
        forbidden.append("禁止15:00后临时操作")
        return FinalDecision(
            action="不动",
            target=None,
            amount_or_ratio="-",
            buy_score=None,
            sell_score=None,
            reason={
                "趋势判断": ["盘后复盘阶段"],
                "位置判断": [],
                "资金判断": [],
                "风险判断": ["15:00 后不再生成当天买入建议"],
                "仓位判断": [],
            },
            forbidden=forbidden,
        )

    if status.is_execute_only:
        forbidden.append("14:57-15:00只允许执行，不建议重新决策")

    if ranking_df is None or ranking_df.empty:
        return FinalDecision(
            action="不动",
            target=None,
            amount_or_ratio="-",
            buy_score=None,
            sell_score=None,
            reason={
                "趋势判断": ["数据不足，无法计算评分"],
                "位置判断": [],
                "资金判断": [],
                "风险判断": ["默认保守：不动"],
                "仓位判断": [],
            },
            forbidden=forbidden,
        )

    df = ranking_df.copy()
    if "buy_score" in df.columns:
        df["buy_score"] = pd.to_numeric(df["buy_score"], errors="coerce")
    if "sell_score" in df.columns:
        df["sell_score"] = pd.to_numeric(df["sell_score"], errors="coerce")
    if "dist_ma20" in df.columns:
        df["dist_ma20"] = pd.to_numeric(df["dist_ma20"], errors="coerce")

    holdings = portfolio.holdings.copy()
    holding_names = set(holdings["fund_name"].astype(str).tolist()) if not holdings.empty else set()

    df["is_holding"] = df["fund_name"].astype(str).isin(holding_names)
    df_hold = df[df["is_holding"]].copy()

    if env.systemic_risk:
        forbidden.append("系统性风险优先防守")

    if not df_hold.empty:
        top_risk = df_hold.sort_values(["sell_score", "buy_score"], ascending=[False, True]).iloc[0]
        top_sell_score = top_risk.get("sell_score")
        if pd.notna(top_sell_score) and float(top_sell_score) >= (65 if env.systemic_risk else 80):
            fund_name = str(top_risk["fund_name"])
            sell_score = float(top_sell_score)
            if sell_score >= 80:
                action = "卖出" if sell_score >= 90 or env.systemic_risk else "减仓"
                ratio = "100%" if action == "卖出" else "50%"
            else:
                action = "减仓"
                ratio = "40%"

            buy_breakdown = top_risk.get("buy_breakdown") or {}
            sell_breakdown = top_risk.get("sell_breakdown") or {}
            reason = {
                "趋势判断": _top_reasons(sell_breakdown.get("趋势破坏分", {})),
                "位置判断": _top_reasons(sell_breakdown.get("高位风险分", {})),
                "资金判断": _top_reasons(sell_breakdown.get("资金流出分", {})),
                "风险判断": _top_reasons(sell_breakdown.get("大盘风险分", {})) or (["系统性风险偏高"] if env.systemic_risk else []),
                "仓位判断": _top_reasons(sell_breakdown.get("持仓风险分", {})),
            }
            forbidden.append("优先处理高卖出分持仓")
            return FinalDecision(
                action=action,
                target=fund_name,
                amount_or_ratio=ratio,
                buy_score=float(top_risk.get("buy_score")) if pd.notna(top_risk.get("buy_score")) else None,
                sell_score=sell_score,
                reason=reason,
                forbidden=forbidden,
            )

    df_buy = df.sort_values(["buy_score", "dist_ma20"], ascending=[False, True]).copy()
    top = df_buy.iloc[0]
    buy_score = top.get("buy_score")
    if pd.isna(buy_score):
        return FinalDecision(
            action="不动",
            target=None,
            amount_or_ratio="-",
            buy_score=None,
            sell_score=None,
            reason={
                "趋势判断": ["评分缺失"],
                "位置判断": [],
                "资金判断": [],
                "风险判断": ["默认保守：不动"],
                "仓位判断": [],
            },
            forbidden=forbidden,
        )

    buy_score_f = float(buy_score)
    if buy_score_f < 70 or (not env.allow_attack and buy_score_f < 85):
        forbidden.append("买入分不足或大盘环境不允许进攻")
        return FinalDecision(
            action="不动",
            target=None,
            amount_or_ratio="-",
            buy_score=buy_score_f,
            sell_score=float(top.get("sell_score")) if pd.notna(top.get("sell_score")) else None,
            reason={
                "趋势判断": ["买入分不够高，继续观察"],
                "位置判断": [],
                "资金判断": [],
                "风险判断": ["默认保守：不动"],
                "仓位判断": [],
            },
            forbidden=forbidden,
        )

    fund_name = str(top["fund_name"])
    track = str(top.get("track") or "")
    dist = float(top.get("dist_ma20")) if pd.notna(top.get("dist_ma20")) else None
    is_holding = bool(top.get("is_holding"))

    buy_amount = _pick_buy_amount(buy_score_f)
    if buy_amount <= 0:
        return FinalDecision(
            action="不动",
            target=None,
            amount_or_ratio="-",
            buy_score=buy_score_f,
            sell_score=float(top.get("sell_score")) if pd.notna(top.get("sell_score")) else None,
            reason={
                "趋势判断": ["买入分不足"],
                "位置判断": [],
                "资金判断": [],
                "风险判断": ["默认保守：不动"],
                "仓位判断": [],
            },
            forbidden=forbidden,
        )

    if dist is not None and dist > 0.08:
        forbidden.append("禁止追高买入：距离20日均线超过8%")
        buy_amount = min(buy_amount, 1000)

    if is_holding:
        holding_row = portfolio.holdings[portfolio.holdings["fund_name"] == fund_name]
        holding_pnl = None
        last_add_date = None
        if not holding_row.empty:
            try:
                holding_pnl = float(holding_row.iloc[0].get("pnl")) if holding_row.iloc[0].get("pnl") not in (None, "") else None
            except Exception:
                holding_pnl = None
            last_add_date = holding_row.iloc[0].get("last_add_date")

        trend_points = 0.0
        buy_breakdown = top.get("buy_breakdown") or {}
        for v in (buy_breakdown.get("趋势分", {}) or {}).values():
            try:
                trend_points += float(v.get("分数", 0))
            except Exception:
                continue
        if holding_pnl is not None and holding_pnl < 0 and trend_points < 18:
            forbidden.append("禁止亏损补仓：趋势分低于18分")
            return FinalDecision(
                action="不动",
                target=None,
                amount_or_ratio="-",
                buy_score=buy_score_f,
                sell_score=float(top.get("sell_score")) if pd.notna(top.get("sell_score")) else None,
                reason={
                    "趋势判断": ["趋势分偏低，禁止因为亏损而补仓"],
                    "位置判断": [],
                    "资金判断": [],
                    "风险判断": ["默认保守：不动"],
                    "仓位判断": [],
                },
                forbidden=forbidden,
            )

    ok, reasons = can_add_more(
        portfolio,
        fund_name=fund_name,
        track=track,
        buy_amount=float(buy_amount),
        today=now.date(),
    )
    if not ok:
        forbidden.extend(reasons)
        return FinalDecision(
            action="不动",
            target=None,
            amount_or_ratio="-",
            buy_score=buy_score_f,
            sell_score=float(top.get("sell_score")) if pd.notna(top.get("sell_score")) else None,
            reason={
                "趋势判断": ["买入分满足，但仓位/现金约束不允许执行"],
                "位置判断": [],
                "资金判断": [],
                "风险判断": ["默认保守：不动"],
                "仓位判断": reasons[:3],
            },
            forbidden=forbidden,
        )

    action = "加仓" if is_holding else "买入"

    buy_breakdown = top.get("buy_breakdown") or {}
    reason = {
        "趋势判断": _top_reasons(buy_breakdown.get("趋势分", {})),
        "位置判断": _top_reasons(buy_breakdown.get("位置分", {})),
        "资金判断": _top_reasons(buy_breakdown.get("资金分", {})),
        "风险判断": _top_reasons(buy_breakdown.get("大盘环境分", {})),
        "仓位判断": [f"当前仓位约 {fund_position_ratio(portfolio, fund_name):.0%}，买入金额 {format_currency(buy_amount)}"],
    }

    return FinalDecision(
        action=action,
        target=fund_name,
        amount_or_ratio=format_currency(buy_amount),
        buy_score=buy_score_f,
        sell_score=float(top.get("sell_score")) if pd.notna(top.get("sell_score")) else None,
        reason=reason,
        forbidden=forbidden,
    )

