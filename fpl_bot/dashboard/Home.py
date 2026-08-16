"""FPL Bot dashboard -- entry point.

Run with: streamlit run fpl_bot/dashboard/Home.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `fpl_bot` imports work

import streamlit as st

from fpl_bot.dashboard.utils import MODEL_LABELS, data_freshness, page_header
from fpl_bot.ensemble.team_state import get_team_state, list_pending_approvals

st.set_page_config(page_title="FPL Bot Dashboard", page_icon="⚽", layout="wide")

page_header("FPL Bot -- System Overview", "Every model's inner workings, one page each. Use the sidebar to dig in.")

fresh = data_freshness()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Players loaded", fresh["n_players"])
c2.metric("Teams loaded", fresh["n_teams"])
c3.metric("Fixtures loaded", fresh["n_fixtures"])
c4.metric("Historical training rows", f"{fresh['n_historical_rows']:,}")

if fresh["next_gw"]:
    gw_id, gw_name, deadline = fresh["next_gw"]
    st.info(f"Next gameweek: **{gw_name}** (GW{gw_id}), deadline {deadline}")
else:
    st.warning("No upcoming gameweek found -- run the bootstrap ingest.")

st.divider()
st.subheader("The 5 teams")

cols = st.columns(5)
for col, key in zip(cols, ["model1", "model2", "model3", "model4", "ensemble"]):
    with col:
        st.markdown(f"**{MODEL_LABELS[key]}**")
        state = get_team_state(key)
        if state is None:
            st.caption("Not drafted yet")
        else:
            st.metric("Squad size", len(state.squad_ids))
            st.caption(f"GW{state.gw} · captain id {state.captain_id}")

pending = list_pending_approvals()
st.divider()
if pending:
    st.warning(f"⚠️ {len(pending)} item(s) awaiting approval -- see the **Approval Queue** page.")
else:
    st.success("No items currently awaiting approval.")

st.divider()
st.subheader("What each page shows")
st.markdown(
    """
- **Data Layer** -- the raw players/teams/fixtures/historical data everything else is built on
- **Model 3 -- Basic Stats** -- the hand-tuned scoring formula and what it currently ranks highest
- **Model 2 -- Statistical** -- the trained predictor's backtest accuracy, predicted-vs-actual, feature importance
- **Model 1 -- Evolutionary** -- the generation-by-generation learning curve, the genome it evolved, and how it did out-of-training
- **Model 4 -- Qualitative** -- the shortlist, sources, and LLM rationale behind its score adjustments
- **Ensemble** -- how the four models' rankings get combined, and where they agree or disagree
- **Teams & Squads** -- the actual recommended squad for all 5 teams right now
- **Approval Queue** -- what's pending, and a working approve/reject flow
"""
)
