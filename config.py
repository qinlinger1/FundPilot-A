from __future__ import annotations

from dataclasses import dataclass


APP_TITLE = "A股场外基金买卖识别指挥台"
PROJECT_NAME = "a_fund_signal_dashboard"


@dataclass(frozen=True)
class FundConfig:
    name: str
    track: str
    fund_code: str | None
    etf_code: str
    benchmark_index: str


DEFAULT_PRINCIPAL_CNY = 45000.0
DEFAULT_CASH_BUFFER_RATIO = 0.20
MAX_SINGLE_FUND_RATIO = 0.25
MAX_SINGLE_TRACK_RATIO = 0.30
MAX_CONSECUTIVE_ADD_DAYS = 2


BUY_AMOUNT_RULES = [
    (85, 3000),
    (80, 2000),
    (70, 1000),
]


FUND_POOL: list[FundConfig] = [
    FundConfig(
        name="广发中证军工ETF联接C",
        track="军工",
        fund_code=None,
        etf_code="512660",
        benchmark_index="sh000300",
    ),
    FundConfig(
        name="银河中证机器人指数A",
        track="机器人",
        fund_code=None,
        etf_code="159770",
        benchmark_index="sh000300",
    ),
    FundConfig(
        name="半导体/芯片基金",
        track="半导体",
        fund_code=None,
        etf_code="512760",
        benchmark_index="sh000300",
    ),
    FundConfig(
        name="创业板基金",
        track="创业板",
        fund_code=None,
        etf_code="159915",
        benchmark_index="sh000300",
    ),
    FundConfig(
        name="沪深300基金",
        track="宽基",
        fund_code=None,
        etf_code="510300",
        benchmark_index="sh000300",
    ),
    FundConfig(
        name="科创板基金",
        track="科创",
        fund_code=None,
        etf_code="588000",
        benchmark_index="sh000300",
    ),
]


MARKET_INDEXES = {
    "沪深300": "sh000300",
    "上证指数": "sh000001",
    "创业板指": "sz399006",
    "科创50": "sh000688",
}

