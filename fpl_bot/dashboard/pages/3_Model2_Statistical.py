import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from fpl_bot.dashboard.utils import explainer, go_grouped_bar, go_grouped_box, go_scatter_by_group, page_header, render_squad_section
from fpl_bot.models.statistical_predictor import (
    MODEL_PATH,
    get_backtest_arrays,
    load_model_and_test_matrix,
    predict_next_gameweek,
)

st.set_page_config(page_title="Model 2 -- Statistical", page_icon="📈", layout="wide")
page_header("Model 2 -- Statistical Predictor", "Gradient-boosted trees, trained on 3 seasons of real gameweek outcomes.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m fpl_bot.models.statistical_predictor` first.")
    st.stop()

explainer(
    """
A `HistGradientBoostingRegressor` (scikit-learn) predicts each player's points for the next gameweek from
features knowable *before* kickoff: rolling 3-game points/minutes, team & opponent rolling scoring rates,
home/away, price, position. Trained on 2023-24 + 2024-25, backtested on the held-out 2025-26 season below --
"held-out" means the model never saw these rows during training, so this is a genuine out-of-sample test.
"""
)

bt = get_backtest_arrays()
POS_COLORS = {"GK": "#E69F00", "DEF": "#0072B2", "MID": "#009E73", "FWD": "#CC79A7"}

st.subheader("Filter the backtest")
c1, c2 = st.columns([1, 2])
with c1:
    pos_filter = st.multiselect("Position", ["GK", "DEF", "MID", "FWD"], default=["GK", "DEF", "MID", "FWD"])
with c2:
    round_range = st.slider("Gameweek range", int(bt["round"].min()), int(bt["round"].max()),
                             (int(bt["round"].min()), int(bt["round"].max())))
bt_f = bt[(bt["position"].isin(pos_filter)) & (bt["round"].between(*round_range))]

model_mae = mean_absolute_error(bt_f["actual"], bt_f["predicted"])
naive_mae = mean_absolute_error(bt_f["actual"], bt_f["naive_baseline"])
improvement = (1 - model_mae / naive_mae) * 100 if naive_mae else 0

st.subheader(f"Backtest on the filtered slice ({len(bt_f):,} player-gameweeks)")
c1, c2, c3 = st.columns(3)
c1.metric("Model MAE", f"{model_mae:.3f} pts")
c2.metric("Naive baseline MAE", f"{naive_mae:.3f} pts")
c3.metric("Improvement", f"{improvement:.1f}%", delta=f"{naive_mae - model_mae:.3f} pts better")

st.subheader("Predicted vs. actual points")
sample = bt_f.sample(min(3000, len(bt_f)), random_state=1) if len(bt_f) else bt_f
if len(sample):
    fig = go_scatter_by_group(sample, "actual", "predicted", "position", categories=["GK", "DEF", "MID", "FWD"],
                               colors=POS_COLORS, hover_cols=["name", "round"])
    max_val = max(sample["actual"].max(), sample["predicted"].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(color="gray", dash="dash"), name="Perfect prediction"))
    st.plotly_chart(fig, use_container_width=True)
    explainer("Dashed line = perfect prediction. FPL points are genuinely noisy (red cards, bonus-point ties, "
              "rotation) so this won't hug the line -- what matters is the trend and how tight the cloud is "
              "relative to the naive baseline's own scatter (try comparing error distributions below).")
else:
    st.info("No rows match the current filter.")

st.subheader("Error over the season (does accuracy drift?)")
by_round = bt_f.groupby("round").apply(lambda g: pd.Series({
    "Model MAE": mean_absolute_error(g["actual"], g["predicted"]),
    "Naive MAE": mean_absolute_error(g["actual"], g["naive_baseline"]),
})).reset_index()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=by_round["round"], y=by_round["Model MAE"], name="Model", line=dict(color="#0072B2")))
fig2.add_trace(go.Scatter(x=by_round["round"], y=by_round["Naive MAE"], name="Naive baseline", line=dict(color="#999999")))
fig2.update_layout(xaxis_title="Gameweek", yaxis_title="MAE that gameweek")
st.plotly_chart(fig2, use_container_width=True)
explainer("If the model's line stayed consistently below the naive baseline's all season, that's evidence "
          "the edge is real and stable, not a fluke concentrated in a few gameweeks.")

st.subheader("Which features actually drive the predictions?")
st.caption("Permutation importance: how much worse predictions get when one feature is shuffled to noise.")
if st.button("Compute feature importance"):
    with st.spinner("Shuffling features and re-predicting..."):
        model, X_test, y_test = load_model_and_test_matrix()
        sample_idx = X_test.sample(min(2000, len(X_test)), random_state=1).index
        X_sample, y_sample = X_test.loc[sample_idx], y_test.loc[sample_idx]
        result = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=1, scoring="neg_mean_absolute_error")
        imp_df = pd.DataFrame({"feature": X_sample.columns, "importance": result.importances_mean}).sort_values("importance", ascending=True)
        st.plotly_chart(go_grouped_bar(imp_df["feature"].tolist(), {"Importance": imp_df["importance"].tolist()}, horizontal=True),
                         use_container_width=True)

st.divider()
st.subheader("Error by position (filtered)")
bt_f = bt_f.copy()
bt_f["abs_error"] = (bt_f["actual"] - bt_f["predicted"]).abs()
if len(bt_f):
    st.plotly_chart(go_grouped_box(bt_f, "position", "abs_error", categories=["GK", "DEF", "MID", "FWD"]), use_container_width=True)

st.divider()
st.subheader("Final squad (live prediction for the next gameweek)")
render_squad_section(predict_next_gameweek(), key_prefix="m2")
