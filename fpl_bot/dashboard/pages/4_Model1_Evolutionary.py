import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fpl_bot.dashboard.utils import explainer, go_grouped_bar, go_heatmap, page_header, render_squad_section
from fpl_bot.models.evolutionary.genome import FEATURES, POSITIONS
from fpl_bot.models.evolutionary.live_apply import predict_current_squad_scores
from fpl_bot.scripts.train_model1 import MODEL_PATH

st.set_page_config(page_title="Model 1 -- Evolutionary", page_icon="🧬", layout="wide")
page_header("Model 1 -- Evolutionary Strategy", "A population of strategies plays historical seasons; the best survive and breed. This is the \"dino game\" one.")

if not MODEL_PATH.exists():
    st.error("No trained genome found. Run `python -m fpl_bot.scripts.train_model1` first (~5 min).")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

history_df = pd.DataFrame(bundle["history"])
genome_snapshots = bundle["genome_snapshots"]
evolved = bundle["test_result"]
baseline = bundle["baseline_result"]

explainer(
    """
Each **genome** is a candidate strategy: a weight per signal per position, plus two thresholds (how big a
score gain has to be to spend a free transfer, or to justify a -4 hit). Each **generation**, every genome in
the population plays a full simulated season against real historical results (the fitness function), the
best-scoring genomes survive, and the next generation is bred from them via crossover + mutation. This is
the direct analogue of the dino-game population getting better at the game, run after run.
"""
)

st.subheader("Generational learning curve (training)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=history_df["generation"], y=history_df["best"], mode="lines+markers", name="Best genome", line=dict(color="#E69F00")))
fig.add_trace(go.Scatter(
    x=pd.concat([history_df["generation"], history_df["generation"][::-1]]),
    y=pd.concat([history_df["best"], history_df["worst"][::-1]]),
    fill="toself", fillcolor="rgba(150,150,150,0.15)", line=dict(color="rgba(0,0,0,0)"),
    name="Population spread (best-worst)", hoverinfo="skip",
))
fig.add_trace(go.Scatter(x=history_df["generation"], y=history_df["avg"], mode="lines+markers", name="Population average", line=dict(color="#999999", dash="dot")))
fig.add_trace(go.Scatter(x=history_df["generation"], y=history_df["worst"], mode="lines", name="Worst genome", line=dict(color="#CC79A7", dash="dot")))
fig.update_layout(xaxis_title="Generation", yaxis_title="Fitness (total points across training seasons, minus hit penalties)")
st.plotly_chart(fig, use_container_width=True)
explainer(
    "The shaded band is the full population's spread each generation -- how it narrows over time tells you "
    "whether the population is converging on a shared strategy or still exploring. A widening 'worst' gap "
    "(the pink dotted line lagging) shows mutation is still injecting genuinely bad genomes even late on, "
    "which is expected and healthy -- it's what keeps the search from getting stuck.",
    label="Why show worst/spread, not just best?",
)

c1, c2, c3 = st.columns(3)
c1.metric("Fitness at generation 0", f"{history_df['best'].iloc[0]:.0f}")
c2.metric("Fitness at final generation", f"{history_df['best'].iloc[-1]:.0f}", delta=f"{history_df['best'].iloc[-1] - history_df['best'].iloc[0]:.0f}")
c3.metric("Population std, final gen", f"{history_df['std'].iloc[-1]:.0f}", help="Lower = population has converged on a similar strategy")

st.divider()
st.subheader("Watch the strategy evolve")
st.caption("Drag through generations to see what the best genome valued at that point in training -- and, further "
           "down, what squad it would have drafted for the *current* live gameweek.")
gen_idx = st.slider("Generation", 0, len(genome_snapshots) - 1, len(genome_snapshots) - 1)
inspected_genome = genome_snapshots[gen_idx]

weights_df = pd.DataFrame({pos: inspected_genome.weights[pos] for pos in POSITIONS}).T
st.plotly_chart(
    go_heatmap(weights_df, title=f"Best genome's weights as of generation {gen_idx} (blue = positive, red = negative)",
               colorscale="RdBu", zmid=0),
    use_container_width=True,
)
c1, c2 = st.columns(2)
c1.metric("Transfer threshold", f"{inspected_genome.transfer_threshold:.3f}", help="Min. score gain to spend a free transfer")
c2.metric("Hit threshold", f"{inspected_genome.hit_threshold:.3f}", help="Min. score gain to justify a -4 hit")

st.divider()
st.subheader("Backtest on held-out 2025-26 season (out-of-training, final genome only)")
st.caption("Both genomes run through the identical simulator on a season neither has seen -- "
           "the baseline is a sensible non-evolved default (trust recent form, nothing else), not a strawman.")

compare_df = pd.DataFrame([
    {"Genome": "Evolved", "Total points": evolved["total_points"], "Raw points": evolved["raw_points"], "Hits taken": evolved["hits_taken"]},
    {"Genome": "Naive baseline", "Total points": baseline["total_points"], "Raw points": baseline["raw_points"], "Hits taken": baseline["hits_taken"]},
])
c1, c2 = st.columns([1, 2])
with c1:
    st.dataframe(compare_df.set_index("Genome"), use_container_width=True)
    diff = evolved["total_points"] - baseline["total_points"]
    st.metric("Evolution's margin", f"{diff:+.0f} pts", help="Over one full season, after hit penalties")
with c2:
    st.plotly_chart(
        go_grouped_bar(compare_df["Genome"].tolist(),
                       {"Total points": compare_df["Total points"].tolist(), "Hits taken": compare_df["Hits taken"].tolist()},
                       colors={"Total points": "#E69F00", "Hits taken": "#999999"}),
        use_container_width=True,
    )

st.subheader("Week-by-week points during the backtest season")
weeks_df = pd.DataFrame({
    "Gameweek": range(1, len(evolved["history"]) + 1),
    "Evolved": evolved["history"],
    "Naive baseline": baseline["history"],
})
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=weeks_df["Gameweek"], y=weeks_df["Evolved"], name="Evolved", line=dict(color="#E69F00")))
fig2.add_trace(go.Scatter(x=weeks_df["Gameweek"], y=weeks_df["Naive baseline"], name="Naive baseline", line=dict(color="#999999")))
fig2.update_layout(xaxis_title="Gameweek", yaxis_title="Points that week")
st.plotly_chart(fig2, use_container_width=True)

st.info(
    "**Known v1 simplifications** (by design, not oversight): at most one transfer evaluated per week, "
    "free-transfer bank caps at 2, no bench autosubs, no chips (wildcard/free hit/bench boost/triple captain)."
)

st.divider()
st.subheader(f"Final squad (using generation {gen_idx}'s genome, applied to the current live gameweek)")
render_squad_section(predict_current_squad_scores(genome=inspected_genome), key_prefix="m1")
