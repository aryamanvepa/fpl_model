import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from fpl_bot.dashboard.utils import (
    MODEL_COLORS,
    MODEL_LABELS,
    explainer,
    get_all_model_scores_cached,
    go_grouped_bar,
    page_header,
    render_squad_section,
)
from fpl_bot.ensemble.combine import MODEL_WEIGHTS, compute_ensemble_scores, model_agreement, percentile_ranks

st.set_page_config(page_title="Ensemble", page_icon="⚖️", layout="wide")
page_header("Ensemble (Team E)", "How the four models' rankings get combined into one decision.")

explainer(
    """
The four models score players in incompatible units (Model 2 = predicted points ~0-8, Model 1 = a
genome-weighted linear score ~-3 to 3, Models 3/4 = a 0-1 composite), so raw scores aren't averaged
directly. Each model's scores are first converted to a **percentile rank within that model** (0 = that
model's worst pick, 1 = its best), and only the ranks get combined with the weights below. If a model's
score is missing for a player (or the whole model failed that run, e.g. Model 4's LLM backend being down),
it's dropped and the remaining weights redistribute automatically.
"""
)

st.subheader("Tune the weights yourself")
st.caption("Defaults reflect what's actually been backtested: Models 1 & 2 both beat a naive baseline on "
           "held-out data, so they start weighted highest. Drag any slider and the tables below recompute live.")

if "ens_weights" not in st.session_state:
    st.session_state.ens_weights = dict(MODEL_WEIGHTS)

if st.button("Reset to defaults"):
    st.session_state.ens_weights = dict(MODEL_WEIGHTS)
    st.rerun()

cols = st.columns(4)
for col, key in zip(cols, ["model1", "model2", "model3", "model4"]):
    with col:
        st.session_state.ens_weights[key] = st.slider(MODEL_LABELS[key], 0.0, 1.0, st.session_state.ens_weights[key], 0.05, key=f"ens_{key}")

current_weights = st.session_state.ens_weights
if sum(current_weights.values()) == 0:
    st.warning("All weights are 0 -- set at least one above 0 to compute an ensemble.")
    st.stop()

weights_df = pd.DataFrame([{"Model": MODEL_LABELS[k], "Weight": w, "key": k} for k, w in current_weights.items()])
fig = go_grouped_bar(weights_df["Model"].tolist(), {"Weight": weights_df["Weight"].tolist()})
fig.update_traces(marker_color=[MODEL_COLORS[k] for k in weights_df["key"]], text=weights_df["Weight"], textposition="outside")
st.plotly_chart(fig, use_container_width=True)

model_result = get_all_model_scores_cached("ollama")
scores = model_result["scores"]

if model_result["errors"]:
    for key, err in model_result["errors"].items():
        st.warning(f"{MODEL_LABELS.get(key, key)} unavailable this run: {err}")

if not scores:
    st.error("No model scores available.")
    st.stop()

ensemble_scores = compute_ensemble_scores(scores, weights=current_weights)
rank_maps = {k: percentile_ranks(v) for k, v in scores.items()}

st.divider()
st.subheader("Top 20 -- and exactly how each model voted")
top20 = ensemble_scores[:20]
rows = []
for p in top20:
    row = {"Name": p.web_name, "Ensemble score": p.score}
    for key in ["model1", "model2", "model3", "model4"]:
        row[MODEL_LABELS[key]] = round(rank_maps.get(key, {}).get(p.player_id, float("nan")), 3)
    rows.append(row)
contrib_df = pd.DataFrame(rows)
column_config = {
    MODEL_LABELS[k]: st.column_config.ProgressColumn(MODEL_LABELS[k], min_value=0, max_value=1, format="%.2f")
    for k in ["model1", "model2", "model3", "model4"]
}
st.dataframe(contrib_df, use_container_width=True, hide_index=True, column_config=column_config)
explainer("Each model column is that player's percentile rank *within that model* (1.0 = that model's #1 pick, "
          "blank = that model didn't score this player, e.g. Model 4 unavailable this run). A player can be an "
          "ensemble favorite even without topping any single model, by ranking well across several.")

st.divider()
st.subheader("Where the models agree")
agreement = model_agreement(scores, top_n=40)
consensus = sorted(((pid, models) for pid, models in agreement.items() if len(models) >= 3), key=lambda x: -len(x[1]))
if consensus:
    name_lookup = {p.player_id: p.web_name for v in scores.values() for p in v}
    st.dataframe(
        pd.DataFrame([{"Name": name_lookup.get(pid, pid), "Models agreeing": len(models),
                        "Which models": ", ".join(MODEL_LABELS[m] for m in models)} for pid, models in consensus]),
        use_container_width=True, hide_index=True,
    )
else:
    st.caption("No player currently sits in 3+ models' individual top 40.")

st.divider()
st.subheader("Final squad (using the weights above)")
render_squad_section(ensemble_scores, key_prefix="ens")
