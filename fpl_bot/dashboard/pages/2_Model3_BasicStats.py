import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from fpl_bot.dashboard.utils import (
    explainer,
    go_grouped_bar,
    go_grouped_violin,
    go_heatmap,
    load_players_with_names,
    page_header,
    render_squad_section,
)
from fpl_bot.models.basic_stats import POSITION_WEIGHTS, compute_scores

st.set_page_config(page_title="Model 3 -- Basic Stats", page_icon="🧮", layout="wide")
page_header("Model 3 -- Basic Stats", "A transparent, hand-tuned formula. No ML -- the fast-to-ship baseline the other models get measured against.")

explainer(
    """
Every player's score is a weighted sum of 5 signals, computed separately per position (a "good defender"
and "good striker" mean different things statistically):

- **pts_rate** -- points per game last season (a rate, not a total, so it doesn't penalize players who missed games)
- **value** -- last season's total points divided by current price (rewards cheap-but-productive players)
- **ict** -- FPL's own Influence/Creativity/Threat blended index
- **attack** -- expected goal involvements (xG + xA): underlying attacking quality, not just actual goals
- **defense** -- *negative* expected goals conceded: underlying defensive quality (only matters for GK/DEF)

The whole thing is then multiplied by an availability factor: 0 if injured/suspended, scaled down if merely doubtful.
"""
)

POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
FEATURE_LABELS = {"pts_rate": "Points rate", "value": "Value (pts/£m)", "ict": "ICT index", "attack": "Attack (xGI)", "defense": "Defense (-xGC)"}

st.subheader("Tune the formula yourself")
st.caption("Drag any slider and every chart and ranking below recomputes live -- this is the whole formula, not a mockup.")

if "m3_weights" not in st.session_state:
    st.session_state.m3_weights = {pos: dict(feats) for pos, feats in POSITION_WEIGHTS.items()}

if st.button("Reset to defaults"):
    st.session_state.m3_weights = {pos: dict(feats) for pos, feats in POSITION_WEIGHTS.items()}
    st.rerun()

cols = st.columns(4)
for col, pos_id in zip(cols, [1, 2, 3, 4]):
    with col:
        st.markdown(f"**{POSITION_NAMES[pos_id]}**")
        for feature in ["pts_rate", "value", "ict", "attack", "defense"]:
            st.session_state.m3_weights[pos_id][feature] = st.slider(
                FEATURE_LABELS[feature], 0.0, 1.0, st.session_state.m3_weights[pos_id][feature], 0.05,
                key=f"m3_{pos_id}_{feature}",
            )

current_weights = st.session_state.m3_weights

st.divider()
st.subheader("Current weights vs. default")
weights_df = pd.DataFrame(current_weights).T
weights_df.index = ["GK", "DEF", "MID", "FWD"]
st.plotly_chart(go_heatmap(weights_df, title="Weight given to each signal, by position (live)", colorscale="Blues"), use_container_width=True)

st.divider()
st.subheader("Do these signals actually track real outcomes?")
players_df = load_players_with_names()
players_df["value_metric"] = players_df.apply(lambda r: r["total_points"] / (r["now_cost"] / 10) if r["now_cost"] else 0, axis=1)
corr_rows = []
for pos_id, pos_name in POSITION_NAMES.items():
    sub = players_df[players_df["element_type"] == pos_id]
    for col, label in [("points_per_game", "pts_rate"), ("value_metric", "value"), ("ict_index", "ict"),
                        ("expected_goal_involvements", "attack"), ("expected_goals_conceded", "defense (raw, unflipped)")]:
        if sub[col].notna().sum() > 2:
            corr_rows.append({"Position": pos_name, "Signal": label, "Correlation with total points": sub[col].corr(sub["total_points"])})
corr_df = pd.DataFrame(corr_rows)
fig = go_grouped_bar(
    [f"{r.Position}/{r.Signal}" for r in corr_df.itertuples()],
    {"Correlation": corr_df["Correlation with total points"].tolist()},
)
st.plotly_chart(fig, use_container_width=True)
explainer(
    """
**Caveat:** `pts_rate` (points per game) is almost tautologically correlated with total points -- it's derived
from it. It's included here as a sanity-check reference point, not evidence the formula is circular: the
formula uses it as one signal among five specifically *because* recent scoring rate is a strong predictor of
future scoring rate, which is a legitimate (if unsurprising) thing to weight heavily.
""",
    label="Why is pts_rate correlation so high?",
)

st.divider()
st.subheader("What it currently ranks highest (with your weights)")
scores = compute_scores(weights_override=current_weights)
scores_df = pd.DataFrame([{"Name": p.web_name, "Position": POSITION_NAMES[p.element_type], "Price": p.now_cost / 10, "Score": p.score} for p in scores])

POS_COLORS = {"GK": "#E69F00", "DEF": "#0072B2", "MID": "#009E73", "FWD": "#CC79A7"}
tabs = st.tabs(["Overall Top 20", "GK", "DEF", "MID", "FWD"])
with tabs[0]:
    top20 = scores_df.sort_values("Score", ascending=True).tail(20)
    fig = go_grouped_bar(top20["Name"].tolist(), {"Score": top20["Score"].tolist()}, horizontal=True)
    fig.update_traces(marker_color=[POS_COLORS[p] for p in top20["Position"]])
    st.plotly_chart(fig, use_container_width=True)
for tab, pos in zip(tabs[1:], ["GK", "DEF", "MID", "FWD"]):
    with tab:
        sub = scores_df[scores_df["Position"] == pos].sort_values("Score", ascending=False).head(15)
        st.dataframe(sub, use_container_width=True, hide_index=True)

st.subheader("Score distribution by position")
st.plotly_chart(go_grouped_violin(scores_df, "Position", "Score", categories=["GK", "DEF", "MID", "FWD"]), use_container_width=True)

st.divider()
st.subheader("Final squad (using the weights above)")
render_squad_section(scores, key_prefix="m3")
