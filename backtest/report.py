"""Generate HTML backtest reports with plotly charts."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from backtest.simulator import BacktestResult

logger = logging.getLogger(__name__)


class BacktestReporter:
    """Produces a self-contained HTML report from backtest results."""

    def __init__(self, output_dir: str = "data/backtest_results") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: BacktestResult,
        filename: str | None = None,
    ) -> str:
        """Create HTML report with plotly charts and summary table.

        Returns the absolute path to the generated file.
        """
        if filename is None:
            ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"report_{ts}.html"

        # --- Summary table ---
        summary_html = self._build_summary_table(result)

        # --- Charts ---
        charts_html = self._build_charts(result)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }}
        h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
        h2 {{ color: #8b949e; margin-top: 30px; }}
        table {{ border-collapse: collapse; margin: 20px 0; width: auto; }}
        th, td {{ border: 1px solid #30363d; padding: 8px 16px; text-align: right; }}
        th {{ background: #161b22; color: #58a6ff; }}
        td {{ background: #0d1117; }}
        .positive {{ color: #3fb950; }}
        .negative {{ color: #f85149; }}
        .chart-container {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Polymarket Backtest Report</h1>
    {summary_html}
    {charts_html}
</body>
</html>"""

        filepath = self._output_dir / filename
        filepath.write_text(html, encoding="utf-8")
        logger.info("Report saved to %s", filepath)
        return str(filepath.resolve())

    def _build_summary_table(self, result: BacktestResult) -> str:
        """Build the HTML summary metrics table."""
        ret_class = "positive" if result.total_return >= 0 else "negative"
        ret_pct = f"{result.total_return * 100:+.2f}%"

        rows = [
            ("Total Return", f'<span class="{ret_class}">{ret_pct}</span>'),
            ("Sharpe Ratio", f"{result.sharpe_ratio:.2f}"),
            ("Max Drawdown", f"{result.max_drawdown * 100:.1f}%"),
            ("Win Rate", f"{result.win_rate * 100:.1f}%"),
            ("Brier Score", f"{result.brier_score:.4f}"),
            ("Avg Edge", f"{result.avg_edge * 100:.2f}%"),
            ("Profit Factor", f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞"),
            ("Total Trades", str(result.n_trades)),
        ]

        row_html = "\n".join(
            f"            <tr><td style='text-align:left;'>{label}</td><td>{value}</td></tr>"
            for label, value in rows
        )
        return f"""
    <h2>Summary</h2>
    <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
{row_html}
        </tbody>
    </table>"""

    def _build_charts(self, result: BacktestResult) -> str:
        """Generate all plotly chart divs."""
        sections: list[str] = []

        # 1. Equity curve
        sections.append(self._chart_equity_curve(result.equity_curve))

        # 2. Drawdown chart
        sections.append(self._chart_drawdown(result.equity_curve))

        if result.n_trades > 0:
            # 3. Rolling win rate
            sections.append(self._chart_rolling_win_rate(result.trades))

            # 4. Edge distribution
            sections.append(self._chart_edge_distribution(result.trades))

            # 5. P&L by category
            sections.append(self._chart_pnl_by_category(result.by_category))

            # 6. Trade scatter
            sections.append(self._chart_trade_scatter(result.trades))

        return "\n".join(sections)

    def _fig_to_html(self, fig: go.Figure) -> str:
        """Convert a plotly figure to an embeddable HTML div."""
        return (
            '<div class="chart-container">'
            + fig.to_html(full_html=False, include_plotlyjs="cdn")
            + "</div>"
        )

    def _chart_equity_curve(self, equity_curve: pd.Series) -> str:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode="lines",
            name="Equity",
            line=dict(color="#58a6ff", width=2),
        ))
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Time",
            yaxis_title="Bankroll (USDC)",
            template="plotly_dark",
            height=400,
        )
        return self._fig_to_html(fig)

    def _chart_drawdown(self, equity_curve: pd.Series) -> str:
        if len(equity_curve) < 2:
            return ""
        cummax = equity_curve.cummax()
        drawdown = (cummax - equity_curve) / cummax

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=-drawdown.values,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#f85149", width=1),
            fillcolor="rgba(248, 81, 73, 0.3)",
        ))
        fig.update_layout(
            title="Drawdown",
            xaxis_title="Time",
            yaxis_title="Drawdown",
            template="plotly_dark",
            height=350,
        )
        return self._fig_to_html(fig)

    def _chart_rolling_win_rate(self, trades: pd.DataFrame) -> str:
        if trades.empty:
            return ""
        wins = (trades["pnl"] > 0).astype(float)
        rolling_wr = wins.rolling(window=20, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(rolling_wr))),
            y=rolling_wr.values,
            mode="lines",
            name="Win Rate (rolling 20)",
            line=dict(color="#3fb950", width=2),
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#8b949e")
        fig.update_layout(
            title="Rolling Win Rate (20-trade window)",
            xaxis_title="Trade #",
            yaxis_title="Win Rate",
            template="plotly_dark",
            height=350,
        )
        return self._fig_to_html(fig)

    def _chart_edge_distribution(self, trades: pd.DataFrame) -> str:
        if trades.empty or "edge" not in trades.columns:
            return ""

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trades["edge"],
            nbinsx=30,
            marker_color="#58a6ff",
            opacity=0.8,
            name="Edge",
        ))
        fig.update_layout(
            title="Edge Distribution at Entry",
            xaxis_title="Edge",
            yaxis_title="Count",
            template="plotly_dark",
            height=350,
        )
        return self._fig_to_html(fig)

    def _chart_pnl_by_category(self, by_category: dict) -> str:
        if not by_category:
            return ""
        categories = list(by_category.keys())
        pnls = [by_category[c]["total_pnl"] for c in categories]
        colors = ["#3fb950" if p >= 0 else "#f85149" for p in pnls]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=pnls,
            marker_color=colors,
            name="P&L",
        ))
        fig.update_layout(
            title="P&L by Category",
            xaxis_title="Category",
            yaxis_title="P&L (USDC)",
            template="plotly_dark",
            height=350,
        )
        return self._fig_to_html(fig)

    def _chart_trade_scatter(self, trades: pd.DataFrame) -> str:
        if trades.empty:
            return ""

        yes_trades = trades[trades["direction"] == "YES"]
        no_trades = trades[trades["direction"] == "NO"]

        fig = go.Figure()
        if not yes_trades.empty:
            fig.add_trace(go.Scatter(
                x=yes_trades["edge"],
                y=yes_trades["pnl"],
                mode="markers",
                name="YES",
                marker=dict(color="#3fb950", size=6, opacity=0.7),
            ))
        if not no_trades.empty:
            fig.add_trace(go.Scatter(
                x=no_trades["edge"],
                y=no_trades["pnl"],
                mode="markers",
                name="NO",
                marker=dict(color="#f85149", size=6, opacity=0.7),
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="#8b949e")
        fig.update_layout(
            title="Trade Scatter: Edge vs P&L",
            xaxis_title="Edge at Entry",
            yaxis_title="P&L (USDC)",
            template="plotly_dark",
            height=400,
        )
        return self._fig_to_html(fig)
