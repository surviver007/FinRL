"""
Generate a lightweight backtest chart using Plotly (HTML output)
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_backtest_result.csv")
df = pd.read_csv(csv_path)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05,
    subplot_titles=("Account Value", "Daily Return %"),
)

# Account value line
fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["account_value"],
        name="PPO",
        line=dict(color="#636EFA", width=2),
        fill="tozeroy",
        fillcolor="rgba(99,110,250,0.08)",
        hovertemplate="Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
    ),
    row=1, col=1,
)

# Initial capital baseline
fig.add_hline(
    y=1000000, line_dash="dash", line_color="gray", line_width=1, opacity=0.5,
    row=1, col=1,
)

# Daily return bars
colors = ["#00CC96" if r >= 0 else "#EF553B" for r in df["daily_return_pct"]]
fig.add_trace(
    go.Bar(
        x=df["date"],
        y=df["daily_return_pct"],
        name="Daily Return",
        marker_color=colors,
        opacity=0.7,
        hovertemplate="Date: %{x}<br>Return: %{y:.3f}%<extra></extra>",
    ),
    row=2, col=1,
)

fig.update_layout(
    title=dict(
        text="PPO Agent Backtest - A-Share (2025.01~2025.05)",
        font=dict(size=18),
    ),
    template="plotly_white",
    height=600,
    width=1100,
    hovermode="x unified",
    showlegend=False,
    margin=dict(l=60, r=30, t=60, b=40),
)

fig.update_yaxes(row=1, col=1, tickprefix="$", tickformat=",")
fig.update_yaxes(row=2, col=1, ticksuffix="%")
fig.update_xaxes(row=2, col=1, tickangle=-45, tickfont=dict(size=9))

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_backtest_chart.html")
fig.write_html(out_path)
print(f"Chart saved: {out_path}")
print("Open it in your browser!")
