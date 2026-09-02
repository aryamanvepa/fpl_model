import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fpl_bot.dashboard.utils import explainer, page_header
from fpl_bot.features.fixtures import fixture_heatmap_grid, player_fixture_features, team_fixture_outlook
from fpl_bot.models.basic_stats import compute_scores

st.set_page_config(page_title="Fixture Heatmap", page_icon="🗓️", layout="wide")
page_header("Fixture Heatmap", "Who has the easiest run coming up -- and which defenders that makes attractive.")

explainer(
    """
FPL publishes a **Fixture Difficulty Rating** (FDR) of 1-5 per fixture, from the perspective of
each team. This grid converts that to an "ease" score where **green = easy, red = hard**, so a
horizontal band of green is a team with a good run.

Two things a plain average would get wrong, handled here:
- **Blank gameweeks** (a team has no fixture that week) show as a gap, not as average difficulty --
  no fixture means guaranteed zero points, which is worse than a hard fixture, not neutral.
- **Double gameweeks** (two fixtures in one week) count as two scoring chances.

The models use this as an `opportunity` score = ease x fixtures-per-gameweek, weighted most
heavily for goalkeepers and defenders, whose clean-sheet points depend far more on the quality
of the opponent than attacking returns do.
""",
    label="How to read this",
)

horizon = st.slider("Gameweeks to look ahead", 3, 10, 5)

team_names, gws, ease_matrix, label_matrix = fixture_heatmap_grid(horizon)

fig = go.Figure(
    data=go.Heatmap(
        z=ease_matrix,
        x=[f"GW{g}" for g in gws],
        y=team_names,
        text=label_matrix,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=[[0.0, "#d73027"], [0.5, "#fee08b"], [1.0, "#1a9850"]],
        zmin=0, zmax=1,
        hovertemplate="%{y} vs %{text}<br>ease=%{z:.2f}<extra></extra>",
        colorbar={"title": "Ease"},
    )
)
fig.update_layout(height=max(420, 26 * len(team_names)), margin=dict(l=60, r=20, t=30, b=40),
                   yaxis={"autorange": "reversed"})
st.plotly_chart(fig, use_container_width=True)
st.caption("Sorted by best overall run first. Cell labels show the opponent and (H)ome / (A)way.")

st.divider()
st.subheader(f"Best defensive picks over the next {horizon} gameweeks")
st.caption("Model 3's score (which already weights fixtures heavily for defenders) alongside the raw "
           "fixture outlook, so you can see whether a defender is rated for their own quality, their "
           "fixtures, or both.")

scores = compute_scores(fixture_horizon=horizon)
fixture_feats = player_fixture_features(horizon)
outlook = team_fixture_outlook(horizon)

from fpl_bot import db  # noqa: E402 -- local import keeps the page's heavy imports at the top

conn = db.get_connection()
try:
    teams = {r[0]: r[1] for r in conn.execute("SELECT id, short_name FROM teams")}
finally:
    conn.close()

defenders = [p for p in scores if p.element_type in (1, 2)]
defenders.sort(key=lambda p: -p.score)
rows = []
for p in defenders[:20]:
    f = fixture_feats.get(p.player_id, {})
    rows.append(
        {
            "Player": p.web_name,
            "Team": teams.get(p.team_id, "?"),
            "Pos": "GKP" if p.element_type == 1 else "DEF",
            "Price": f"£{p.now_cost / 10:.1f}m",
            "Model 3 score": round(p.score, 3),
            "Fixture ease": round(f.get("ease", 0), 2),
            "Fixtures": f.get("n_fixtures", 0),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.info(
    "**Note on the word 'heatmap':** this is a fixture-difficulty heatmap (the standard FPL fixture "
    "ticker). Player *positional* heatmaps -- where a player physically touches the ball on the pitch -- "
    "aren't available from the FPL API and would need a separate data source such as Understat or FBref."
)
