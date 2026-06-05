import hmac
import html
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

st.set_page_config(page_title="🔮 Polymarket Neural Trader", layout="wide")

st.error(
    "⚠️ EXPERIMENTAL SOFTWARE — Trading prediction markets carries substantial risk of loss. "
    "This software is provided AS-IS with no warranty. You may lose all funds. "
    "Not financial advice. You are solely responsible for compliance with your local laws "
    "(including Polymarket's geo-restrictions on US users)."
)


def _check_password() -> bool:
    """Return True if the user has entered the correct password."""
    dashboard_password = os.environ.get("DASHBOARD_PASSWORD", "")
    if not dashboard_password or dashboard_password == "changeme":
        st.error(
            "🔐 DASHBOARD_PASSWORD is not set or is the default 'changeme'. "
            "Set a strong password in your .env file to access the dashboard."
        )
        st.stop()

    def _verify():
        entered = st.session_state.get("password_input", "")
        if hmac.compare_digest(entered, dashboard_password):
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
            st.error("Incorrect password.")

    if st.session_state.get("authenticated"):
        return True

    st.subheader("🔐 Dashboard Login")
    st.text_input("Password", type="password", key="password_input", on_change=_verify)
    st.stop()


_check_password()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "trades.db"
CONFIG_PATH = BASE_DIR / "config" / "settings.yaml"
MODEL_PATH = BASE_DIR / "data" / "models" / "tcn_latest.pt"

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return yaml.safe_load(CONFIG_PATH.read_text()) or {}
        except Exception:
            return {}
    return {}


@st.cache_data(ttl=60)
def load_portfolio_stats() -> dict:
    if not DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT COALESCE(SUM(pnl), 0.0) FROM trades WHERE pnl IS NOT NULL")
        total_pnl: float = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM trades WHERE status IN ('closed', 'resolved') AND pnl IS NOT NULL")
        closed: int = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM trades WHERE status IN ('closed', 'resolved') AND pnl > 0")
        wins: int = cur.fetchone()[0]
        win_rate = wins / closed if closed > 0 else 0.0

        cur.execute("SELECT COUNT(*) FROM trades WHERE status='open'")
        open_positions: int = cur.fetchone()[0]

        cur.execute("SELECT pnl FROM trades WHERE pnl IS NOT NULL ORDER BY id ASC")
        pnl_rows = cur.fetchall()
        cumulative = 0.0
        peak = 0.0
        for r in pnl_rows:
            cumulative += r[0]
            if cumulative > peak:
                peak = cumulative
        current_drawdown = (peak - cumulative) / max(peak, 1e-8) if peak > 0.0 else 0.0

        conn.close()
        return {
            "total_pnl": round(total_pnl, 4),
            "win_rate": round(win_rate, 4),
            "open_positions": open_positions,
            "current_drawdown": round(max(current_drawdown, 0.0), 4),
        }
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_pnl_curve() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(
            "SELECT timestamp, pnl FROM trades WHERE pnl IS NOT NULL ORDER BY id ASC",
            conn,
        )
        conn.close()
        if df.empty:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["cumulative_pnl"] = df["pnl"].cumsum()
        return df[["timestamp", "cumulative_pnl"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_recent_trades(n: int = 20) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(
            """
            SELECT id, market_id, question, direction,
                   entry_price, size_usdc, estimated_prob, edge_at_entry AS edge,
                   regime_at_entry AS regime, status, exit_price, pnl,
                   category, is_paper, timestamp
            FROM trades ORDER BY id DESC LIMIT ?
            """,
            conn,
            params=(n,),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_brier_scores() -> dict:
    if not DB_PATH.exists():
        return {}
    try:
        import json

        import numpy as np

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT estimated_prob, signal_breakdown, direction, exit_price, pnl
            FROM trades WHERE status IN ('closed', 'resolved') AND pnl IS NOT NULL
            ORDER BY id DESC LIMIT 50
            """
        )
        rows = cur.fetchall()
        conn.close()

        overall_errors: list[float] = []
        signal_errors: dict[str, list[float]] = {}

        for row in rows:
            pnl_val = row["pnl"] or 0.0
            direction = row["direction"]
            if pnl_val > 0:
                actual = 1.0 if direction == "YES" else 0.0
            elif pnl_val < 0:
                actual = 0.0 if direction == "YES" else 1.0
            else:
                continue  # skip zero pnl trades
            if row["estimated_prob"] is not None:
                overall_errors.append((float(row["estimated_prob"]) - actual) ** 2)
            try:
                weights = json.loads(row["signal_breakdown"] or "{}")
                total_w = sum(abs(float(w)) for w in weights.values()) or 1.0
                for sig, w in weights.items():
                    share = abs(float(w)) / total_w
                    signal_errors.setdefault(sig, []).append(share)
            except Exception:
                logger.debug("Skipping malformed signal breakdown", exc_info=True)

        return {
            "overall": float(np.mean(overall_errors)) if overall_errors else None,
            "per_signal": {
                sig: round(float(sum(errs) / len(errs)), 6)
                for sig, errs in signal_errors.items()
                if errs
            },
        }
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_current_regime() -> str:
    if not DB_PATH.exists():
        return "unknown"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT regime_at_entry AS regime FROM trades ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        return row["regime"] if row and row["regime"] else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Helper: regime badge
# ---------------------------------------------------------------------------

_REGIME_COLORS = {
    "stable": ("🟢", "#22c55e"),
    "trending": ("🟡", "#eab308"),
    "volatile": ("🔴", "#ef4444"),
}


def _regime_badge(regime: str) -> None:
    icon, color = _REGIME_COLORS.get(regime.lower(), ("⚪", "#94a3b8"))
    st.markdown(
        f'<span style="font-size:1.4rem;font-weight:700;color:{color}">'
        f'{icon} {regime.upper()}</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    dashboard_password = os.environ.get("DASHBOARD_PASSWORD")
    if dashboard_password:
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if not st.session_state.authenticated:
            st.title("🔮 Polymarket Neural Trader")
            password = st.text_input("Enter dashboard password:", type="password")
            if st.button("Login"):
                import hmac
                if hmac.compare_digest(password, dashboard_password):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid password")
            return

    cfg = load_config()
    stats = load_portfolio_stats()
    regime = load_current_regime()

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        st.metric("Bankroll", f"${cfg.get('bankroll', '—')}")
        dry_run = cfg.get("dry_run", True)
        st.markdown(
            f"**Mode:** {'🧪 Dry Run' if dry_run else '💰 Live'}",
        )
        st.markdown("**Regime:** ", unsafe_allow_html=False)
        _regime_badge(regime)
        st.divider()
        st.caption(f"Min edge: {cfg.get('min_edge', 0.08):.0%}")
        st.caption(f"Kelly fraction: {cfg.get('kelly_fraction', 0.25):.0%}")
        st.caption(f"Max positions: {cfg.get('max_positions', 10)}")
        st.caption(f"Max drawdown stop: {cfg.get('drawdown_stop', 0.15):.0%}")
        st.divider()
        now_str = datetime.now(UTC).strftime("%H:%M:%S UTC")
        st.caption(f"🕒 {now_str}")

    # ── Title ─────────────────────────────────────────────────────
    st.title("🔮 Polymarket Neural Trader")

    # ── Tabs ──────────────────────────────────────────────────────
    tab_portfolio, tab_markets, tab_signals, tab_analytics = st.tabs(
        ["📊 Portfolio", "🔍 Markets", "🧠 Signals & Model", "📈 Analytics"]
    )

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — Portfolio
    # ══════════════════════════════════════════════════════════════
    with tab_portfolio:
        c1, c2, c3, c4 = st.columns(4)
        total_pnl = stats.get("total_pnl", 0.0)
        pnl_delta = f"${total_pnl:+.2f}" if total_pnl != 0 else None
        c1.metric("Total P&L", f"${total_pnl:.2f}", delta=pnl_delta)
        c2.metric("Win Rate", f"{stats.get('win_rate', 0.0) * 100:.1f}%")
        c3.metric("Open Positions", stats.get("open_positions", 0))
        drawdown = stats.get("current_drawdown", 0.0)
        c4.metric(
            "Current Drawdown",
            f"{drawdown:.1%}",
            delta=f"{-drawdown:.1%}" if drawdown > 0 else None,
            delta_color="inverse",
        )

        st.divider()

        # P&L curve
        st.subheader("P&L Curve")
        pnl_curve = load_pnl_curve()
        if not pnl_curve.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=pnl_curve["timestamp"],
                    y=pnl_curve["cumulative_pnl"],
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color="#6366f1", width=2),
                    fillcolor="rgba(99,102,241,0.15)",
                    name="Cumulative P&L",
                )
            )
            fig.update_layout(
                hovermode="x unified",
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="P&L (USDC)",
                xaxis_title="",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No resolved trades yet — P&L curve will appear here.")

        st.divider()

        # Recent trades table
        st.subheader("Recent Trades (Last 20)")
        trades_df = load_recent_trades(20)
        if trades_df.empty:
            st.info("No trades recorded yet.")
        else:
            display = trades_df.copy()
            if "question" in display.columns:
                display["question"] = display["question"].fillna("").astype(str).str[:60].map(html.escape)
            if "edge" in display.columns:
                display["edge"] = (display["edge"] * 100).round(2).astype(str) + "%"
            if "status" in display.columns:
                display["status_display"] = display["status"].map(
                    {"open": "🟡 Open", "closed": "✅ Closed"}
                ).fillna("❓ Unknown")
            st.dataframe(display, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — Markets
    # ══════════════════════════════════════════════════════════════
    with tab_markets:
        col_btn, col_status = st.columns([1, 4])
        with col_btn:
            do_refresh = st.button("🔄 Refresh Markets", use_container_width=True)
        with col_status:
            if "markets_loaded_at" in st.session_state:
                st.caption(f"Last scanned: {st.session_state.markets_loaded_at}")

        if do_refresh or "scanned_markets" not in st.session_state:
            with st.spinner("Scanning markets…"):
                try:
                    import asyncio
                    import concurrent.futures
                    import sys
                    sys.path.insert(0, str(BASE_DIR / "src"))
                    from polymarket_agent.data.gamma_client import GammaClient

                    async def _scan_markets():
                        gamma = GammaClient()
                        try:
                            return await gamma.get_all_markets()
                        finally:
                            await gamma.close()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        raw_markets = pool.submit(asyncio.run, _scan_markets()).result()
                    markets = [
                        {
                            "question": m.question,
                            "outcome_prices": [],
                            "yes_price": 0.0,
                            "volume_24h": m.volume_24h,
                            "liquidity": m.liquidity,
                            "days_to_resolution": (
                                (m.end_date - datetime.now(UTC)).days
                                if m.end_date else 0
                            ),
                            "condition_id": m.condition_id,
                        }
                        for m in raw_markets
                    ]
                    st.session_state.scanned_markets = sorted(
                        markets, key=lambda m: m.get("liquidity", 0), reverse=True
                    )
                    st.session_state.markets_loaded_at = datetime.now(UTC).strftime(
                        "%Y-%m-%d %H:%M:%S UTC"
                    )
                except Exception as exc:
                    st.error(f"Scan failed: {exc}")
                    st.session_state.scanned_markets = []

        markets_list: list[dict] = st.session_state.get("scanned_markets", [])
        if not markets_list:
            st.info("Click **Refresh Markets** to scan for candidates.")
        else:
            min_edge = cfg.get("min_edge", 0.08)
            rows = []
            for m in markets_list:
                prices = m.get("outcome_prices", [])
                yes_price = prices[0] if prices else m.get("yes_price", 0.0)
                mid = float(yes_price) if yes_price else 0.0
                # Simple order-book edge estimate vs market mid
                edge = abs(mid - 0.5) * 0.4 if mid > 0 else 0.0

                row: dict = {
                    "question": (m.get("question") or "")[:70],
                    "yes_price": f"{mid:.4f}",
                    "volume_24h": f"${m.get('volume_24h', 0):,.0f}",
                    "liquidity": f"${m.get('liquidity', 0):,.0f}",
                    "days_to_resolution": round(m.get("days_to_resolution", 0), 1),
                }
                if edge >= min_edge:
                    row["edge"] = f"{edge:.1%} ✅"
                else:
                    row["edge"] = "—"
                rows.append(row)

            markets_df = pd.DataFrame(rows)
            st.dataframe(markets_df, use_container_width=True, hide_index=True)
            st.caption(f"{len(markets_list)} markets found, sorted by liquidity ↓")

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — Signals & Model
    # ══════════════════════════════════════════════════════════════
    with tab_signals:
        col_left, col_right = st.columns(2)

        # Signal weights bar chart
        with col_left:
            st.subheader("Signal Weights")
            weights = cfg.get("signal_weights", {})
            if weights:
                wdf = pd.DataFrame(
                    {"signal": list(weights.keys()), "weight": list(weights.values())}
                ).sort_values("weight", ascending=True)
                fig_w = go.Figure(
                    go.Bar(
                        x=wdf["weight"],
                        y=wdf["signal"],
                        orientation="h",
                        marker_color="#6366f1",
                    )
                )
                fig_w.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=280,
                    xaxis_title="Weight",
                )
                st.plotly_chart(fig_w, use_container_width=True)
            else:
                st.info("No signal weights in config.")

        # Signal contribution chart
        with col_right:
            st.subheader("Signal Contribution Share")
            brier = load_brier_scores()
            per_signal = brier.get("per_signal", {})
            if per_signal:
                bdf = pd.DataFrame(
                    {"signal": list(per_signal.keys()), "share": list(per_signal.values())}
                ).sort_values("share", ascending=True)
                fig_b = go.Figure(
                    go.Bar(
                        x=bdf["share"],
                        y=bdf["signal"],
                        orientation="h",
                        marker_color=[
                            "#22c55e" if v < 0.20 else "#eab308" if v < 0.25 else "#ef4444"
                            for v in bdf["share"]
                        ],
                    )
                )
                fig_b.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=280,
                    xaxis_title="Contribution Share",
                )
                st.plotly_chart(fig_b, use_container_width=True)
                if brier.get("overall") is not None:
                    st.caption(f"Overall Brier Score: **{brier['overall']:.4f}**")
            else:
                st.info("No calibration data yet (need resolved trades).")

        st.divider()

        col_regime, col_tcn = st.columns(2)

        # Regime indicator
        with col_regime:
            st.subheader("Current Regime")
            _regime_badge(regime)
            st.caption("Derived from most recent trade's detected regime.")

        # TCN model status
        with col_tcn:
            st.subheader("TCN Model")
            if MODEL_PATH.exists():
                mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
                st.success(f"✅ Loaded — last trained {mtime.strftime('%Y-%m-%d')}")
            else:
                st.warning("⚠️ Not loaded — run `python scripts/train_tcn.py` to train.")
            # Show torch availability
            try:
                import torch  # noqa: F401
                st.caption("PyTorch: available")
            except ImportError:
                st.caption("PyTorch: not installed")


    # ══════════════════════════════════════════════════════════════
    # TAB 4 — Analytics
    # ══════════════════════════════════════════════════════════════
    with tab_analytics:
        st.header("Portfolio Analytics")

        analytics_query = """
            SELECT id, timestamp, direction, entry_price, exit_price, pnl,
                   size_usdc, status, category, edge_at_entry
            FROM trades
            WHERE pnl IS NOT NULL
            ORDER BY id ASC
        """
        if not DB_PATH.exists():
            analytics_df = pd.DataFrame()
        else:
            try:
                conn = sqlite3.connect(str(DB_PATH))
                analytics_df = pd.read_sql_query(analytics_query, conn)
                conn.close()
            except Exception:
                analytics_df = pd.DataFrame()

        if analytics_df.empty:
            st.info("No completed trades yet. Analytics will appear once trades are closed.")
        else:
            # Cumulative P&L chart
            analytics_df["cumulative_pnl"] = analytics_df["pnl"].cumsum()
            analytics_df["trade_number"] = range(1, len(analytics_df) + 1)

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=analytics_df["trade_number"],
                y=analytics_df["cumulative_pnl"],
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color="cyan", width=2),
                marker=dict(size=4),
            ))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_pnl.update_layout(
                title="Cumulative P&L Over Time",
                xaxis_title="Trade #",
                yaxis_title="Cumulative P&L ($)",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

            # Win rate over time (rolling) + P&L distribution
            col1, col2 = st.columns(2)

            with col1:
                analytics_df["is_win"] = (analytics_df["pnl"] > 0).astype(int)
                analytics_df["rolling_win_rate"] = (
                    analytics_df["is_win"].expanding().mean() * 100
                )

                fig_wr = go.Figure()
                fig_wr.add_trace(go.Scatter(
                    x=analytics_df["trade_number"],
                    y=analytics_df["rolling_win_rate"],
                    mode="lines",
                    name="Win Rate %",
                    line=dict(color="lime", width=2),
                ))
                fig_wr.add_hline(y=50, line_dash="dash", line_color="yellow")
                fig_wr.update_layout(
                    title="Rolling Win Rate",
                    xaxis_title="Trade #",
                    yaxis_title="Win Rate (%)",
                    template="plotly_dark",
                    height=350,
                )
                st.plotly_chart(fig_wr, use_container_width=True)

            with col2:
                fig_dist = go.Figure()
                wins = analytics_df[analytics_df["pnl"] > 0]["pnl"]
                losses = analytics_df[analytics_df["pnl"] <= 0]["pnl"]
                if not wins.empty:
                    fig_dist.add_trace(go.Histogram(
                        x=wins, name="Wins", marker_color="green", opacity=0.7,
                    ))
                if not losses.empty:
                    fig_dist.add_trace(go.Histogram(
                        x=losses, name="Losses", marker_color="red", opacity=0.7,
                    ))
                fig_dist.update_layout(
                    title="P&L Distribution",
                    xaxis_title="P&L ($)",
                    yaxis_title="Count",
                    template="plotly_dark",
                    barmode="overlay",
                    height=350,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            # Category breakdown
            if "category" in analytics_df.columns and analytics_df["category"].notna().any():
                st.subheader("Performance by Category")
                cat_stats = analytics_df.groupby("category").agg(
                    trades=("pnl", "count"),
                    total_pnl=("pnl", "sum"),
                    avg_pnl=("pnl", "mean"),
                    win_rate=("is_win", "mean"),
                ).round(2)
                cat_stats["win_rate"] = (cat_stats["win_rate"] * 100).round(1)
                cat_stats.columns = ["Trades", "Total P&L ($)", "Avg P&L ($)", "Win Rate (%)"]
                st.dataframe(cat_stats, use_container_width=True)

            # Key metrics summary
            st.subheader("Key Metrics")
            total_pnl = analytics_df["pnl"].sum()
            win_count = (analytics_df["pnl"] > 0).sum()
            loss_count = (analytics_df["pnl"] <= 0).sum()
            total = len(analytics_df)
            avg_win = (
                analytics_df[analytics_df["pnl"] > 0]["pnl"].mean() if win_count > 0 else 0
            )
            avg_loss = (
                analytics_df[analytics_df["pnl"] <= 0]["pnl"].mean() if loss_count > 0 else 0
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total P&L", f"${total_pnl:.2f}")
            m2.metric(
                "Win Rate",
                f"{win_count / total * 100:.1f}%" if total > 0 else "N/A",
            )
            m3.metric("Avg Win", f"${avg_win:.2f}")
            m4.metric("Avg Loss", f"${avg_loss:.2f}")
            profit_factor = (
                abs(avg_win * win_count / (avg_loss * loss_count))
                if loss_count > 0 and avg_loss != 0
                else float("inf")
            )
            m5.metric(
                "Profit Factor",
                f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞",
            )

            # Drawdown chart
            analytics_df["max_pnl"] = analytics_df["cumulative_pnl"].expanding().max()
            analytics_df["drawdown"] = (
                analytics_df["cumulative_pnl"] - analytics_df["max_pnl"]
            )

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=analytics_df["trade_number"],
                y=analytics_df["drawdown"],
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="red", width=1),
                fillcolor="rgba(255,0,0,0.2)",
            ))
            fig_dd.update_layout(
                title="Drawdown from Peak",
                xaxis_title="Trade #",
                yaxis_title="Drawdown ($)",
                template="plotly_dark",
                height=300,
            )
            st.plotly_chart(fig_dd, use_container_width=True)


if __name__ == "__main__":
    main()
