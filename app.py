from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import config
from data_fetcher import fetch_etf_daily, fetch_etf_spot, fetch_index_daily, fetch_open_fund_nav
from db import connect, default_db_path, init_db, load_decisions, load_holdings, save_decision, upsert_holdings
from decision import decide
from portfolio import HOLDING_COLUMNS, compute_portfolio_state, fund_position_ratio
from strategy import compute_indicators, compute_market_env, score_buy, score_sell
from utils import format_currency, format_pct, get_trading_status, now_cn


st.set_page_config(page_title=config.APP_TITLE, layout="wide")


@st.cache_data(ttl=600)
def cached_etf_daily(etf_code: str, start: date) -> tuple[pd.DataFrame | None, str | None]:
    r = fetch_etf_daily(etf_code, start=start)
    return r.df, r.error


@st.cache_data(ttl=20)
def cached_etf_spot(etf_code: str) -> tuple[pd.DataFrame | None, str | None]:
    r = fetch_etf_spot(etf_code)
    return r.df, r.error


@st.cache_data(ttl=1800)
def cached_index_daily(index_symbol: str, start: date) -> tuple[pd.DataFrame | None, str | None]:
    r = fetch_index_daily(index_symbol, start=start)
    return r.df, r.error


@st.cache_data(ttl=3600)
def cached_fund_nav(fund_code: str) -> tuple[pd.DataFrame | None, str | None]:
    r = fetch_open_fund_nav(fund_code)
    return r.df, r.error


def _default_fund_pool_df() -> pd.DataFrame:
    rows = []
    for f in config.FUND_POOL:
        rows.append(
            {
                "fund_name": f.name,
                "track": f.track,
                "fund_code": f.fund_code or "",
                "etf_code": f.etf_code,
                "benchmark_index": f.benchmark_index,
            }
        )
    return pd.DataFrame(rows)


def _style_ranking(df: pd.DataFrame):
    def color_score(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 80:
            return "background-color:#1b5e20;color:white;"
        if x >= 70:
            return "background-color:#2e7d32;color:white;"
        if x >= 60:
            return "background-color:#f9a825;color:black;"
        return "background-color:#b71c1c;color:white;"

    def color_action(v):
        s = str(v)
        if "卖出" in s or "减仓" in s:
            return "background-color:#b71c1c;color:white;"
        if "加仓" in s or "买入" in s:
            return "background-color:#1b5e20;color:white;"
        if "小买" in s:
            return "background-color:#f9a825;color:black;"
        return "background-color:#424242;color:white;"

    return (
        df.style.applymap(color_score, subset=["买入分", "卖出分"])
        .applymap(color_action, subset=["建议动作"])
        .format(
            {
                "近5日涨幅": "{:.2%}",
                "近10日涨幅": "{:.2%}",
                "近20日涨幅": "{:.2%}",
                "距离20日均线": "{:.2%}",
            }
            ,
            na_rep="-",
        )
    )


def _build_action_hint(is_holding: bool, buy_score: float | None, sell_score: float | None) -> str:
    b = -1 if buy_score is None else float(buy_score)
    s = -1 if sell_score is None else float(sell_score)
    if is_holding and s >= 80:
        return "卖出/减仓"
    if is_holding and 65 <= s < 80:
        return "减仓"
    if b >= 80:
        return "加仓" if is_holding else "买入"
    if 70 <= b < 80:
        return "小买观察"
    return "不动"


def _extract_spot_fields(spot_row: pd.DataFrame | None) -> dict:
    if spot_row is None or spot_row.empty:
        return {"ok": False}
    r = spot_row.iloc[0].to_dict()

    def pick(cols: list[str]):
        for c in cols:
            if c in r:
                return r.get(c)
        for k in r.keys():
            if str(k).strip() in cols:
                return r.get(k)
        return None

    price = pick(["最新价", "现价", "最新", "价格", "close", "最新价(元)"])
    pct = pick(["涨跌幅", "涨跌幅(%)", "涨跌幅%", "pct_chg"])
    amount = pick(["成交额", "成交额(元)", "amount"])

    try:
        price_f = float(price) if price not in (None, "") else None
    except Exception:
        price_f = None

    try:
        pct_f = float(pct) if pct not in (None, "") else None
    except Exception:
        pct_f = None

    if pct_f is not None and abs(pct_f) > 1.5:
        pct_f = pct_f / 100.0

    try:
        amount_f = float(amount) if amount not in (None, "") else None
    except Exception:
        amount_f = None

    if price_f is None and pct_f is None and amount_f is None:
        return {"ok": False}
    return {"ok": True, "price": price_f, "pct_chg": pct_f, "amount": amount_f}


def _render_final_decision_block(d) -> str:
    target = d.target or "-"
    lines = []
    lines.append("【今日最终动作】")
    lines.append(f"动作：{d.action}")
    lines.append(f"对象：{target}")
    lines.append(f"金额/比例：{d.amount_or_ratio}")
    lines.append("理由：")
    lines.append("1. 趋势判断：")
    for x in d.reason.get("趋势判断", [])[:5]:
        lines.append(f"   - {x}")
    lines.append("2. 位置判断：")
    for x in d.reason.get("位置判断", [])[:5]:
        lines.append(f"   - {x}")
    lines.append("3. 资金判断：")
    for x in d.reason.get("资金判断", [])[:5]:
        lines.append(f"   - {x}")
    lines.append("4. 风险判断：")
    for x in d.reason.get("风险判断", [])[:5]:
        lines.append(f"   - {x}")
    lines.append("5. 仓位判断：")
    for x in d.reason.get("仓位判断", [])[:5]:
        lines.append(f"   - {x}")
    lines.append("")
    lines.append("【禁止动作】")
    for x in d.forbidden[:12]:
        lines.append(f"- {x}")
    return "\n".join(lines)


def main():
    st.title(config.APP_TITLE)

    decision_placeholder = st.empty()

    if "fund_pool_df" not in st.session_state:
        st.session_state["fund_pool_df"] = _default_fund_pool_df()

    if "principal" not in st.session_state:
        st.session_state["principal"] = float(config.DEFAULT_PRINCIPAL_CNY)

    db_path = default_db_path()
    conn = connect(db_path)
    init_db(conn)

    holdings_df = load_holdings(conn)
    status = get_trading_status(now_cn())

    with st.sidebar:
        st.subheader("设置")
        st.session_state["principal"] = st.number_input(
            "当前总本金（元）",
            min_value=0.0,
            value=float(st.session_state["principal"]),
            step=1000.0,
        )
        start_days = st.slider("拉取历史数据天数", min_value=60, max_value=400, value=180, step=10)
        start_date = date.today() - timedelta(days=int(start_days))
        use_spot = st.toggle("盘中尝试实时ETF行情(不稳定)", value=True)
        if st.button("清除缓存并刷新"):
            st.cache_data.clear()
            st.rerun()

        st.subheader("基金池配置（v1：会话内生效）")
        st.session_state["fund_pool_df"] = st.data_editor(
            st.session_state["fund_pool_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "fund_name": st.column_config.TextColumn("基金名称", required=True),
                "track": st.column_config.TextColumn("赛道", required=True),
                "fund_code": st.column_config.TextColumn("基金代码(可空)"),
                "etf_code": st.column_config.TextColumn("信号源ETF代码", required=True),
                "benchmark_index": st.column_config.TextColumn("基准指数代码", required=True),
            },
        )
        st.download_button(
            "下载当前基金池(JSON)",
            data=json.dumps(st.session_state["fund_pool_df"].to_dict(orient="records"), ensure_ascii=False, indent=2),
            file_name="fund_pool.json",
        )

    portfolio = compute_portfolio_state(float(st.session_state["principal"]), holdings_df)

    idx_map = {}
    idx_errors = {}
    for name, symbol in config.MARKET_INDEXES.items():
        df, err = cached_index_daily(symbol, start_date)
        if df is not None:
            idx_map[name] = df
        if err:
            idx_errors[name] = err

    env = compute_market_env(idx_map)

    head_col1, head_col2, head_col3 = st.columns([2.2, 1.6, 1.2])

    with head_col1:
        st.subheader("总览区")
        s1, s2 = st.columns([1.4, 1.0])
        s1.metric("北京时间", status.now.strftime("%Y-%m-%d %H:%M:%S"))
        s2.metric("交易阶段", status.phase)
        s3, s4, s5 = st.columns(3)
        s3.metric("交易日(今日)", "是" if status.is_trade_day else "否")
        s4.metric("盘中", "是" if status.is_intraday else "否")
        s5.metric("终判(14:50-14:57)", "是" if status.is_final_window else "否")
        if env.systemic_risk:
            st.error("风险：系统性风险偏高（优先防守）")
        elif not env.allow_attack:
            st.warning("风险：大盘环境一般（默认谨慎）")
        else:
            st.success("风险：大盘环境允许进攻（仍需看个体评分）")

    with head_col2:
        st.subheader("资金与持仓")
        m1, m2 = st.columns(2)
        m1.metric("当前总本金", format_currency(portfolio.principal))
        m2.metric("可用现金(估算)", format_currency(portfolio.cash))
        st.metric("持仓市值(估算)", format_currency(portfolio.holdings_value))
        st.caption("现金=本金-持仓市值（v1估算，不含申购/赎回在途资金）。")

        with st.expander("当前基金持仓明细", expanded=False):
            if portfolio.holdings.empty:
                st.info("暂无持仓（因此现金=本金属于正常现象）")
            else:
                show_h = portfolio.holdings.copy()
                show_h = show_h.rename(
                    columns={
                        "fund_name": "基金名称",
                        "track": "赛道",
                        "position_value": "持仓金额(元)",
                        "pnl": "盈亏比例",
                        "allow_add": "允许加仓",
                        "last_add_date": "上次加仓日期",
                    }
                )
                keep = ["基金名称", "赛道", "持仓金额(元)", "盈亏比例", "允许加仓", "上次加仓日期"]
                keep = [c for c in keep if c in show_h.columns]
                st.dataframe(show_h[keep], use_container_width=True, height=180)

    with head_col3:
        st.subheader("今日最终动作")
        st.caption("每次只输出一个主动作。")
        decision_placeholder.info("正在计算中…")

    if idx_errors:
        st.warning("部分大盘指数数据获取失败：" + "；".join([f"{k}:{v}" for k, v in idx_errors.items()]))

    st.divider()

    st.subheader("强弱排序区")

    fund_pool_df = st.session_state["fund_pool_df"].copy()
    fund_pool_df = fund_pool_df[fund_pool_df["fund_name"].astype(str).str.strip() != ""].reset_index(drop=True)
    if fund_pool_df.empty:
        st.info("基金池为空，请在左侧添加。")
        return

    ranking_rows = []
    detail_cache = {}
    for _, row in fund_pool_df.iterrows():
        fund_name = str(row.get("fund_name") or "").strip()
        etf_code = str(row.get("etf_code") or "").strip()
        bench_symbol = str(row.get("benchmark_index") or "").strip()
        track = str(row.get("track") or "").strip()
        fund_code = str(row.get("fund_code") or "").strip()

        etf_df, etf_err = cached_etf_daily(etf_code, start_date)
        bench_df = None
        bench_err = None
        if bench_symbol:
            bench_df, bench_err = cached_index_daily(bench_symbol, start_date)

        if etf_df is None:
            ranking_rows.append(
                {
                    "fund_name": fund_name,
                    "track": track,
                    "etf_code": etf_code,
                    "信号日期": None,
                    "口径": "失败",
                    "近5日涨幅": None,
                    "近10日涨幅": None,
                    "近20日涨幅": None,
                    "距离20日均线": None,
                    "买入分": None,
                    "卖出分": None,
                    "建议动作": "数据获取失败",
                    "错误": etf_err or "数据获取失败",
                }
            )
            continue

        ind = compute_indicators(etf_df, bench_df)
        signal_date = (
            pd.to_datetime(etf_df["date"], errors="coerce").dropna().max().date()
            if "date" in etf_df.columns and not etf_df.empty
            else None
        )
        basis = "日K"

        spot_used = False
        if use_spot and status.is_intraday:
            spot_df, _ = cached_etf_spot(etf_code)
            spot = _extract_spot_fields(spot_df)
            if spot.get("ok"):
                if spot.get("price") is not None:
                    ind["close"] = float(spot["price"])
                    if ind.get("ma20") is not None:
                        ind["dist_ma20"] = float(ind["close"] / ind["ma20"] - 1.0)
                        ind["above_ma20"] = ind["close"] >= ind["ma20"]
                if spot.get("pct_chg") is not None:
                    ind["pct_chg"] = float(spot["pct_chg"])
                if spot.get("amount") is not None:
                    ind["amount"] = float(spot["amount"])
                spot_used = True
                basis = "实时"

        holding_ratio = fund_position_ratio(portfolio, fund_name)
        holding_pnl = None
        if not portfolio.holdings.empty:
            r2 = portfolio.holdings[portfolio.holdings["fund_name"] == fund_name]
            if not r2.empty and r2.iloc[0].get("pnl") not in (None, ""):
                try:
                    holding_pnl = float(r2.iloc[0]["pnl"])
                except Exception:
                    holding_pnl = None

        buy_score, buy_breakdown = score_buy(ind, env)
        sell_score, sell_breakdown = score_sell(ind, env, holding_position_ratio=holding_ratio, holding_pnl=holding_pnl)

        is_holding = fund_name in set(portfolio.holdings["fund_name"].astype(str).tolist()) if not portfolio.holdings.empty else False
        hint = _build_action_hint(is_holding, buy_score, sell_score)

        ranking_rows.append(
            {
                "fund_name": fund_name,
                "track": track,
                "etf_code": etf_code,
                "信号日期": signal_date,
                "口径": basis,
                "近5日涨幅": ind.get("etf_ret_5"),
                "近10日涨幅": ind.get("etf_ret_10"),
                "近20日涨幅": ind.get("etf_ret_20"),
                "距离20日均线": ind.get("dist_ma20"),
                "买入分": buy_score,
                "卖出分": sell_score,
                "建议动作": hint,
                "错误": etf_err or bench_err,
            }
        )
        detail_cache[fund_name] = {
            "etf_df": etf_df,
            "bench_df": bench_df,
            "ind": ind,
            "buy_breakdown": buy_breakdown,
            "sell_breakdown": sell_breakdown,
            "track": track,
            "fund_code": fund_code,
            "etf_code": etf_code,
            "signal_date": signal_date,
            "basis": basis,
            "spot_used": spot_used,
        }

    ranking_df = pd.DataFrame(ranking_rows)
    show_cols = [
        "fund_name",
        "track",
        "etf_code",
        "信号日期",
        "口径",
        "近5日涨幅",
        "近10日涨幅",
        "近20日涨幅",
        "距离20日均线",
        "买入分",
        "卖出分",
        "建议动作",
    ]
    rank_show = ranking_df[show_cols].rename(
        columns={"fund_name": "基金名称", "track": "赛道", "etf_code": "ETF代码"}
    )
    st.dataframe(_style_ranking(rank_show), use_container_width=True, height=320)

    st.divider()

    st.subheader("单基金详情区")
    fund_options = [r["fund_name"] for r in ranking_rows if r.get("fund_name")]
    selected = st.selectbox("选择基金", fund_options, index=0 if fund_options else None)
    if selected and selected in detail_cache:
        info = detail_cache[selected]
        etf_df = info["etf_df"]
        ind = info["ind"]
        buy_breakdown = info["buy_breakdown"]
        sell_breakdown = info["sell_breakdown"]
        signal_date = info.get("signal_date")
        basis = info.get("basis")

        ddf = etf_df.copy()
        ddf = ddf.dropna(subset=["date", "close"]).copy()
        ddf["ma20"] = pd.to_numeric(ddf["close"], errors="coerce").rolling(20).mean()
        chart_df = ddf.set_index("date")[["close", "ma20"]].dropna(how="all")
        st.line_chart(chart_df, height=260)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("近5日涨幅", format_pct(ind.get("etf_ret_5")))
        m2.metric("近10日涨幅", format_pct(ind.get("etf_ret_10")))
        m3.metric("近20日涨幅", format_pct(ind.get("etf_ret_20")))
        m4.metric("距离20日均线", format_pct(ind.get("dist_ma20")))
        m5.metric("信号口径", str(basis or "-"))
        st.caption(f"信号日期：{signal_date or '-'}（口径=实时时，涨跌幅/最新价优先用实时行情，其余指标仍基于历史日K）")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**买入评分拆解**")
            st.json(buy_breakdown, expanded=False)
        with c2:
            st.markdown("**卖出评分拆解**")
            st.json(sell_breakdown, expanded=False)

    st.divider()

    st.subheader("持仓管理区")
    holdings_edit = holdings_df.copy()
    if holdings_edit.empty:
        holdings_edit = pd.DataFrame(columns=HOLDING_COLUMNS)
    holdings_edit = st.data_editor(
        holdings_edit,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "fund_name": st.column_config.TextColumn("基金名称", required=True),
            "fund_code": st.column_config.TextColumn("基金代码"),
            "track": st.column_config.TextColumn("赛道"),
            "etf_code": st.column_config.TextColumn("ETF代码"),
            "position_value": st.column_config.NumberColumn("持仓金额(元)", min_value=0.0, step=100.0),
            "cost": st.column_config.NumberColumn("成本(可空)", step=0.01),
            "pnl": st.column_config.NumberColumn("当前盈亏比例(如0.05)", step=0.01),
            "holding_days": st.column_config.NumberColumn("持有天数", min_value=0, step=1),
            "allow_add": st.column_config.SelectboxColumn("是否允许加仓", options=[0, 1]),
            "last_add_date": st.column_config.TextColumn("上次加仓日期(YYYY-MM-DD)"),
        },
    )
    if st.button("保存持仓"):
        upsert_holdings(conn, holdings_edit)
        st.success("持仓已保存")
        st.rerun()

    st.divider()

    st.subheader("今日最终动作")
    ranking_for_decision = ranking_df.copy()
    ranking_for_decision["buy_score"] = ranking_for_decision["买入分"]
    ranking_for_decision["sell_score"] = ranking_for_decision["卖出分"]
    ranking_for_decision["dist_ma20"] = ranking_for_decision["距离20日均线"]
    ranking_for_decision["buy_breakdown"] = ranking_for_decision["fund_name"].map(
        lambda x: detail_cache.get(x, {}).get("buy_breakdown")
    )
    ranking_for_decision["sell_breakdown"] = ranking_for_decision["fund_name"].map(
        lambda x: detail_cache.get(x, {}).get("sell_breakdown")
    )

    d = decide(status=status, env=env, ranking_df=ranking_for_decision, portfolio=portfolio, now=status.now)
    st.code(_render_final_decision_block(d), language="text")
    decision_placeholder.success(f"动作：{d.action} ｜ 对象：{d.target or '-'} ｜ 金额/比例：{d.amount_or_ratio}")

    save_col1, save_col2 = st.columns([1, 4])
    with save_col1:
        if st.button("保存今日结果"):
            save_decision(
                conn,
                dt=status.now,
                action=d.action,
                target_fund=d.target,
                amount_or_ratio=d.amount_or_ratio,
                buy_score=d.buy_score,
                sell_score=d.sell_score,
                reason=d.reason,
                forbidden=d.forbidden,
                meta={"phase": status.phase, "systemic_risk": env.systemic_risk},
            )
            st.success("已保存到决策记录")
    with save_col2:
        st.caption("提示：14:57-15:00 建议只执行，不重新决策；15:00 后只做复盘，不再生成当天买入建议。")

    st.divider()

    st.subheader("决策记录区")
    hist = load_decisions(conn, limit=200)
    if not hist.empty:
        st.dataframe(hist, use_container_width=True, height=260)
    else:
        st.info("暂无决策记录。")

    st.divider()

    st.subheader("复盘模块（v1：简化）")
    if status.is_after_close:
        if selected and selected in detail_cache:
            info = detail_cache[selected]
            fund_code = info.get("fund_code") or ""
            if fund_code.strip():
                nav_df, nav_err = cached_fund_nav(fund_code.strip())
                if nav_df is None:
                    st.warning(f"基金净值获取失败：{nav_err}")
                else:
                    last_nav_pct = nav_df["pct_chg"].dropna().iloc[-1] if nav_df["pct_chg"].dropna().shape[0] else None
                    etf_df = info["etf_df"]
                    last_etf_pct = etf_df["pct_chg"].dropna().iloc[-1] if etf_df["pct_chg"].dropna().shape[0] else None
                    st.write(
                        {
                            "基金": selected,
                            "ETF日涨跌幅(最近交易日)": float(last_etf_pct) if last_etf_pct is not None else None,
                            "基金净值涨跌幅(最新披露)": float(last_nav_pct) if last_nav_pct is not None else None,
                            "一致性(简化)": "一致" if (last_etf_pct is not None and last_nav_pct is not None and abs(last_etf_pct - last_nav_pct) <= 0.01) else "不一致/未知",
                        }
                    )
            else:
                st.info("该基金未填写基金代码，无法拉取净值。可在左侧基金池配置中补充。")
        else:
            st.info("请选择一只基金进行复盘。")
    else:
        st.info("盘后(15:00后)才展示复盘对比。")


if __name__ == "__main__":
    main()
