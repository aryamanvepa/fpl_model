import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from fpl_bot.dashboard.utils import MODEL_LABELS, get_all_model_scores_cached, page_header
from fpl_bot.ensemble.combine import compute_ensemble_scores
from fpl_bot.optimizer.squad_optimizer import build_squad_result

st.set_page_config(page_title="Teams & Squads", page_icon="👥", layout="wide")
page_header("Teams & Squads", "The actual recommended squad for all 5 teams, right now.")

POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

model_result = get_all_model_scores_cached("ollama")
scores = model_result["scores"]
if scores:
    scores = {**scores, "ensemble": compute_ensemble_scores(scores)}

if model_result["errors"]:
    for key, err in model_result["errors"].items():
        st.caption(f"⚠️ {MODEL_LABELS.get(key, key)} unavailable: {err}")

tabs = st.tabs([MODEL_LABELS.get(k, k) for k in ["model1", "model2", "model3", "model4", "ensemble"]])
for tab, key in zip(tabs, ["model1", "model2", "model3", "model4", "ensemble"]):
    with tab:
        if key not in scores:
            st.info("No data for this team this run.")
            continue
        result = build_squad_result(scores[key])
        c1, c2, c3 = st.columns(3)
        c1.metric("Squad cost", f"£{result.total_cost / 10:.1f}m / £100.0m")
        c2.metric("Captain", result.captain.web_name if result.captain else "-")
        c3.metric("Vice-captain", result.vice_captain.web_name if result.vice_captain else "-")

        def to_df(players, tag_ids=()):
            return pd.DataFrame([
                {"Name": p.web_name + (" (C)" if p.player_id == (result.captain.player_id if result.captain else None) else
                                        " (VC)" if p.player_id == (result.vice_captain.player_id if result.vice_captain else None) else ""),
                 "Pos": POSITION_NAMES[p.element_type], "Price": f"£{p.now_cost/10:.1f}m", "Score": p.score}
                for p in players
            ])

        st.markdown("**Starting XI**")
        st.dataframe(to_df(result.starting_xi), use_container_width=True, hide_index=True)
        st.markdown("**Bench**")
        st.dataframe(to_df(result.bench), use_container_width=True, hide_index=True)
