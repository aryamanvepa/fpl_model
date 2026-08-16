import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from fpl_bot.dashboard.utils import explainer, page_header, render_squad_section
from fpl_bot.models.basic_stats import compute_scores
from fpl_bot.qualitative import sources
from fpl_bot.qualitative.synthesizer import build_shortlist, run_qualitative_review

st.set_page_config(page_title="Model 4 -- Qualitative", page_icon="🤖", layout="wide")
page_header("Model 4 -- Qualitative Agent", "Reads news + injury status for a shortlist of players, an LLM reasons about adjustments.")

explainer(
    """
A single-pass "read this bundle, answer in JSON" call, not a multi-turn agent that decides what to search
next -- simpler and cheaper for a fixed daily source list. Only a shortlist gets reviewed (top scorers per
position from Model 3, plus anyone with an official injury/status note), not all ~570 players, so it stays
fast regardless of which backend answers. The LLM's adjustment (capped at ±30%) is layered on top of
Model 3's base score for reviewed players; everyone else keeps their Model 3 score unchanged.
"""
)

st.subheader("What it reads before deciding anything")
base_scores = compute_scores()
shortlist = build_shortlist(base_scores)
st.caption(f"Shortlist: top scorers per position from Model 3, plus anyone with an official injury/status note "
           f"-- {len(shortlist)} players reviewed, not all ~570 (keeps it fast and cheap regardless of backend).")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Shortlisted players**")
    st.dataframe(
        pd.DataFrame([{"Name": p.web_name, "Position": {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}[p.element_type],
                        "Model 3 score": p.score} for p in shortlist]).sort_values("Model 3 score", ascending=False),
        use_container_width=True, height=350, hide_index=True,
    )
with c2:
    st.markdown("**Recent BBC Sport football headlines**")
    try:
        headlines = sources.fetch_bbc_football_headlines()
        st.dataframe(pd.DataFrame(headlines)[["title", "summary"]], use_container_width=True, height=350, hide_index=True)
    except Exception as e:
        st.error(f"Couldn't fetch headlines: {e}")

st.divider()
st.subheader("Run a live review")
st.caption("This makes a real call to the selected backend. Ollama needs to be running locally; "
           "the cloud backend needs ANTHROPIC_API_KEY set. Nothing runs until you click the button.")

backend = st.radio("Backend", ["ollama", "anthropic"], horizontal=True)
if st.button("Run qualitative review now"):
    with st.spinner(f"Calling {backend}..."):
        try:
            st.session_state.m4_review = run_qualitative_review(backend, base_scores=base_scores)
            st.session_state.m4_error = None
        except RuntimeError as e:
            st.session_state.m4_review = None
            st.session_state.m4_error = str(e)

if st.session_state.get("m4_error"):
    st.error(st.session_state.m4_error)

review = st.session_state.get("m4_review")
if review:
    st.success(f"Got a response. {len(review['notes'])} player(s) adjusted.")
    if review["overall_notes"]:
        st.info(review["overall_notes"])
    if review["notes"]:
        st.dataframe(
            pd.DataFrame([{"Name": n.web_name, "Adjustment": f"{n.adjustment:+.0%}", "Rationale": n.rationale}
                          for n in review["notes"]]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("No adjustments this run -- nothing in the headlines matched a shortlisted player.")

    st.divider()
    st.subheader("Final squad (Model 3 base scores + this review's adjustments)")
    render_squad_section(review["scores"], key_prefix="m4")
else:
    st.caption("Run a review above to see this model's final squad.")
