"""Shared helpers for every dashboard page: cached data loaders and a
consistent color per model so the same team means the same color on every
chart across the whole app.
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fpl_bot import db

# Chart helpers built on plotly.graph_objects rather than plotly.express:
# express eagerly imports xarray (for px.imshow) at package-init time, and
# this machine's xarray build is incompatible with numpy 2.x -- breaks any
# `import plotly.express` at all, not just imshow calls. graph_objects has
# no such dependency, so these are the only charting primitives used here.


def go_grouped_box(df: pd.DataFrame, x: str, y: str, categories: list[str] | None = None,
                    colors: dict | None = None, title: str = "") -> go.Figure:
    fig = go.Figure()
    cats = categories or sorted(df[x].dropna().unique())
    for cat in cats:
        sub = df[df[x] == cat]
        color = colors.get(cat) if colors else None
        fig.add_trace(go.Box(y=sub[y], name=str(cat), marker_color=color))
    fig.update_layout(title=title, showlegend=False)
    return fig


def go_grouped_violin(df: pd.DataFrame, x: str, y: str, categories: list[str] | None = None, title: str = "") -> go.Figure:
    fig = go.Figure()
    cats = categories or sorted(df[x].dropna().unique())
    for cat in cats:
        sub = df[df[x] == cat]
        fig.add_trace(go.Violin(y=sub[y], name=str(cat), box_visible=True, meanline_visible=True))
    fig.update_layout(title=title, showlegend=False)
    return fig


def go_grouped_bar(x_labels, series: dict, title: str = "", horizontal: bool = False,
                    colors: dict | None = None) -> go.Figure:
    fig = go.Figure()
    for name, values in series.items():
        color = colors.get(name) if colors else None
        if horizontal:
            fig.add_trace(go.Bar(y=x_labels, x=values, name=name, orientation="h", marker_color=color))
        else:
            fig.add_trace(go.Bar(x=x_labels, y=values, name=name, marker_color=color))
    fig.update_layout(title=title, barmode="group", showlegend=len(series) > 1)
    return fig


def go_heatmap(matrix_df: pd.DataFrame, title: str = "", colorscale: str = "Blues", zmid=None) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.values,
            x=list(matrix_df.columns),
            y=[str(i) for i in matrix_df.index],
            text=matrix_df.values,
            texttemplate="%{text:.2f}",
            colorscale=colorscale,
            zmid=zmid,
        )
    )
    fig.update_layout(title=title)
    return fig


def go_scatter_by_group(df: pd.DataFrame, x: str, y: str, group: str, categories: list[str] | None = None,
                         colors: dict | None = None, title: str = "", hover_cols: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    cats = categories or sorted(df[group].dropna().unique())
    hover_cols = hover_cols or []
    for cat in cats:
        sub = df[df[group] == cat]
        color = colors.get(cat) if colors else None
        customdata = sub[hover_cols].values if hover_cols else None
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode="markers", name=str(cat), marker=dict(color=color, opacity=0.4),
            customdata=customdata,
            hovertemplate=(", ".join(f"{c}=%{{customdata[{i}]}}" for i, c in enumerate(hover_cols)) + "<extra></extra>")
            if hover_cols else None,
        ))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
    return fig

MODEL_COLORS = {
    "model1": "#E69F00",  # evolutionary strategy -- orange
    "model2": "#0072B2",  # statistical predictor -- blue
    "model3": "#009E73",  # basic stats -- green
    "model4": "#CC79A7",  # qualitative agent -- pink
    "ensemble": "#444444",  # combined -- dark gray, deliberately distinct
}
MODEL_LABELS = {
    "model1": "Model 1 -- Evolutionary",
    "model2": "Model 2 -- Statistical",
    "model3": "Model 3 -- Basic Stats",
    "model4": "Model 4 -- Qualitative",
    "ensemble": "Ensemble (Team E)",
}
POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]


@st.cache_data(ttl=300)
def load_table(name: str) -> pd.DataFrame:
    conn = db.get_connection()
    try:
        return pd.read_sql_query(f"SELECT * FROM {name}", conn)
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_players_with_names() -> pd.DataFrame:
    conn = db.get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT p.*, t.short_name AS team_name, pos.short_name AS position_name
            FROM players p
            JOIN teams t ON p.team_id = t.id
            JOIN positions pos ON p.element_type = pos.id
            """,
            conn,
        )
    finally:
        conn.close()
    return df


def data_freshness() -> dict:
    conn = db.get_connection()
    try:
        n_players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        n_teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
        n_fixtures = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
        n_hist = conn.execute("SELECT COUNT(*) FROM historical_gw").fetchone()[0]
        gw = conn.execute("SELECT id, name, deadline_time FROM gameweeks WHERE is_next = 1").fetchone()
    finally:
        conn.close()
    return {
        "n_players": n_players,
        "n_teams": n_teams,
        "n_fixtures": n_fixtures,
        "n_historical_rows": n_hist,
        "next_gw": gw,
    }


@st.cache_data(ttl=60, show_spinner="Computing all 4 models (+ ensemble)...")
def get_all_model_scores_cached(qualitative_backend: str = "ollama") -> dict:
    """Streamlit reruns the whole script on every widget interaction --
    without caching, switching tabs would recompute all 5 teams from
    scratch (including an Ollama connection attempt) every single time."""
    from fpl_bot.ensemble.models_registry import compute_all_model_scores

    return compute_all_model_scores(qualitative_backend=qualitative_backend)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def explainer(markdown_text: str, label: str = "How to read this") -> None:
    """A collapsed-by-default 'how to read this' note under a chart --
    cheap to add, directly answers 'I need to understand this better'."""
    with st.expander(label):
        st.markdown(markdown_text)


POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def render_squad_section(scores, key_prefix: str = "") -> None:
    """The model's actual final squad -- starting XI, bench, captain/VC,
    cost -- rendered inline on that model's own page, not just the shared
    Teams & Squads page."""
    from fpl_bot.optimizer.squad_optimizer import build_squad_result

    if not scores:
        st.info("No scores available to build a squad from right now.")
        return

    result = build_squad_result(scores)

    c1, c2, c3 = st.columns(3)
    c1.metric("Squad cost", f"£{result.total_cost / 10:.1f}m / £100.0m")
    c2.metric("Captain", result.captain.web_name if result.captain else "-")
    c3.metric("Vice-captain", result.vice_captain.web_name if result.vice_captain else "-")

    def to_rows(players):
        return [
            {
                "Name": p.web_name + (
                    " (C)" if result.captain and p.player_id == result.captain.player_id else
                    " (VC)" if result.vice_captain and p.player_id == result.vice_captain.player_id else ""
                ),
                "Pos": POSITION_NAMES[p.element_type],
                "Price": f"£{p.now_cost / 10:.1f}m",
                "Score": p.score,
            }
            for p in players
        ]

    cxi, cbench = st.columns([2, 1])
    with cxi:
        st.markdown("**Starting XI**")
        st.dataframe(pd.DataFrame(to_rows(result.starting_xi)), use_container_width=True, hide_index=True)
    with cbench:
        st.markdown("**Bench**")
        st.dataframe(pd.DataFrame(to_rows(result.bench)), use_container_width=True, hide_index=True)
