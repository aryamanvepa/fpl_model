"""Renders the dashboard as a static site (charts stay interactive -- zoom,
hover, pan -- via embedded Plotly.js, but nothing recomputes on new data
until this script is run again). GitHub Pages has no Python backend, so this
is how the same views get published: run this locally whenever you want the
public snapshot to reflect the latest models/data, then push.

Reuses the exact chart-building functions the live Streamlit dashboard uses
(fpl_bot/dashboard/utils.py's go_* helpers, and each model's own compute
functions) so the static pages are never a re-implementation, just a
different shell around the same logic.

Run with: python -m fpl_bot.scripts.export_static_site
Output: ./snapshot/ (index.html + one page per dashboard tab)
"""

import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from fpl_bot.dashboard.utils import (
    MODEL_COLORS,
    MODEL_LABELS,
    POSITION_NAMES,
    data_freshness,
    go_grouped_bar,
    go_grouped_box,
    go_grouped_violin,
    go_heatmap,
    go_scatter_by_group,
    load_players_with_names,
    load_table,
)
from fpl_bot.ensemble.combine import MODEL_WEIGHTS, compute_ensemble_scores, model_agreement, percentile_ranks
from fpl_bot.ensemble.models_registry import compute_all_model_scores
from fpl_bot.ensemble.team_state import get_team_state, list_pending_approvals
from fpl_bot.models.basic_stats import POSITION_WEIGHTS, compute_scores
from fpl_bot.models.evolutionary.genome import FEATURES, POSITIONS
from fpl_bot.models.evolutionary.live_apply import predict_current_squad_scores
from fpl_bot.models.statistical_predictor import (
    MODEL_PATH as M2_MODEL_PATH,
    get_backtest_arrays,
    load_model_and_test_matrix,
    predict_next_gameweek,
)
from fpl_bot.optimizer.squad_optimizer import SquadResult, build_squad_result
from fpl_bot.qualitative import sources
from fpl_bot.qualitative.llm_backends import call_ollama
from fpl_bot.qualitative.synthesizer import build_shortlist
from fpl_bot.scripts.train_model1 import MODEL_PATH as M1_MODEL_PATH

OUT_DIR = Path(__file__).resolve().parents[2] / "snapshot"

NAV = [
    ("index.html", "Home"),
    ("data_layer.html", "Data Layer"),
    ("model3.html", "Model 3 – Basic Stats"),
    ("model2.html", "Model 2 – Statistical"),
    ("model1.html", "Model 1 – Evolutionary"),
    ("model4.html", "Model 4 – Qualitative"),
    ("ensemble.html", "Ensemble"),
    ("fixtures.html", "Fixture Heatmap"),
    ("teams.html", "Teams & Squads"),
    ("approvals.html", "Approval Queue"),
]

CSS = """
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh; background-color: #f7f6f2; color: #000;
  font-family: Arial, Helvetica, sans-serif; font-size: 1rem; line-height: 1.55;
}
.wrap { max-width: 68rem; margin: 0 auto; padding: 1.5rem 2rem 4rem; }
h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.15rem; margin: 2.25rem 0 0.9rem; border-top: 1px solid #ccc; padding-top: 1.25rem; }
.subtitle { color: #555; margin: 0 0 1.5rem; font-size: 0.95rem; }
p { margin: 0 0 1rem; }
a { color: #000; }
a:hover { text-decoration: none; }
nav { background: #eae8e1; padding: 0.75rem 2rem; display: flex; flex-wrap: wrap; gap: 0 1.25rem; }
nav a { text-decoration: none; font-size: 0.92rem; padding: 0.25rem 0; white-space: nowrap; }
nav a.active { font-weight: bold; text-decoration: underline; }
.banner { background: #fff3cd; border-bottom: 1px solid #e0c14d; padding: 0.6rem 2rem; font-size: 0.88rem; color: #6b5900; }
.metrics { display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 0 0 1.5rem; }
.metric { min-width: 9rem; }
.metric .label { color: #666; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em; }
.metric .value { font-size: 1.4rem; font-weight: bold; }
table { width: 100%; border-collapse: collapse; margin: 0 0 1.5rem; font-size: 0.92rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem 0.4rem 0; border-bottom: 1px solid #ddd; vertical-align: top; }
th { color: #555; font-weight: normal; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
.note { background: #eae8e1; padding: 0.85rem 1.1rem; border-radius: 4px; margin: 0 0 1.25rem; font-size: 0.92rem; }
.chart { margin: 0 0 1.5rem; }
.cols { display: flex; gap: 2rem; flex-wrap: wrap; }
.cols > div { flex: 1; min-width: 18rem; }
.gen-note { color: #888; font-size: 0.8rem; margin-top: 3rem; border-top: 1px solid #ddd; padding-top: 1rem; }
"""

PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'


def chart_html(fig, height=420) -> str:
    fig.update_layout(height=height, margin=dict(l=40, r=20, t=40, b=40))
    return f'<div class="chart">{fig.to_html(include_plotlyjs=False, full_html=False)}</div>'


def metric(label: str, value: str) -> str:
    return f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>'


def page_shell(active_file: str, title: str, subtitle: str, body: str, banner: str = "") -> str:
    nav_html = "".join(
        f'<a href="{href}" class="{"active" if href == active_file else ""}">{label}</a>' for href, label in NAV
    )
    banner_html = f'<div class="banner">{banner}</div>' if banner else ""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — FPL Multi-Model Bot</title>
{PLOTLY_CDN}
<style>{CSS}</style>
</head>
<body>
<nav>{nav_html}</nav>
{banner_html}
<div class="wrap">
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>
{body}
<div class="gen-note">Static snapshot generated {generated} -- charts are interactive (zoom/hover) but won't recompute until this is regenerated locally and republished. <a href="https://github.com/aryamanvepa/fpl_model">Source</a></p>
</div>
</body>
</html>"""


def squad_section_html(scores) -> str:
    if not scores:
        return '<div class="note">No scores available.</div>'
    result: SquadResult = build_squad_result(scores)
    cap = result.captain.web_name if result.captain else "-"
    vc = result.vice_captain.web_name if result.vice_captain else "-"

    def rows(players):
        out = "<table><tr><th>Name</th><th>Pos</th><th>Price</th><th>Score</th></tr>"
        for p in players:
            tag = " (C)" if result.captain and p.player_id == result.captain.player_id else (
                " (VC)" if result.vice_captain and p.player_id == result.vice_captain.player_id else "")
            out += f"<tr><td>{p.web_name}{tag}</td><td>{POSITION_NAMES[p.element_type]}</td><td>£{p.now_cost/10:.1f}m</td><td>{p.score:.3f}</td></tr>"
        return out + "</table>"

    return (
        f'<div class="metrics">{metric("Squad cost", f"£{result.total_cost/10:.1f}m")}{metric("Captain", cap)}{metric("Vice-captain", vc)}</div>'
        f'<div class="cols"><div><strong>Starting XI</strong>{rows(result.starting_xi)}</div>'
        f'<div><strong>Bench</strong>{rows(result.bench)}</div></div>'
    )


# ---------------------------------------------------------------- Home

def build_home(all_scores: dict) -> str:
    fresh = data_freshness()
    gw = fresh["next_gw"]
    gw_html = f'<div class="note">Next gameweek: <strong>{gw[1]}</strong> (GW{gw[0]}), deadline {gw[2]}</div>' if gw else ""

    body = (
        f'<div class="metrics">{metric("Players loaded", fresh["n_players"])}{metric("Teams loaded", fresh["n_teams"])}'
        f'{metric("Fixtures loaded", fresh["n_fixtures"])}{metric("Historical training rows", f"{fresh["n_historical_rows"]:,}")}</div>'
        + gw_html
        + "<h2>The 5 teams</h2><table><tr><th>Team</th><th>Squad</th><th>Captain</th></tr>"
    )
    for key in ["model1", "model2", "model3", "model4", "ensemble"]:
        state = get_team_state(key)
        if state is None:
            body += f"<tr><td>{MODEL_LABELS[key]}</td><td colspan='2'>Not drafted yet locally</td></tr>"
        else:
            body += f"<tr><td>{MODEL_LABELS[key]}</td><td>{len(state.squad_ids)} players</td><td>id {state.captain_id}</td></tr>"
    body += "</table>"

    pending = list_pending_approvals()
    body += f'<div class="note">{len(pending)} item(s) awaiting approval locally.</div>' if pending else '<div class="note">No items currently awaiting approval.</div>'

    body += (
        "<h2>What each page shows</h2><ul>"
        "<li><strong>Data Layer</strong> -- the raw players/teams/fixtures/historical data everything is built on</li>"
        "<li><strong>Model 3</strong> -- the hand-tuned scoring formula and what it currently ranks highest</li>"
        "<li><strong>Model 2</strong> -- backtest accuracy, predicted-vs-actual, feature importance</li>"
        "<li><strong>Model 1</strong> -- the generation-by-generation learning curve and the genome it evolved</li>"
        "<li><strong>Model 4</strong> -- the shortlist, sources, and (if run) LLM rationale behind score adjustments</li>"
        "<li><strong>Ensemble</strong> -- how the four models' rankings get combined, and where they agree</li>"
        "<li><strong>Teams &amp; Squads</strong> / <strong>Approval Queue</strong> -- all 5 teams and what's pending</li>"
        "</ul>"
    )
    return page_shell("index.html", "FPL Bot -- System Overview", "Every model's inner workings, one page each.", body)


# ---------------------------------------------------------------- Data Layer

def build_data_layer() -> str:
    df = load_players_with_names()
    fig1 = go_grouped_box(df, "position_name", "total_points", categories=["GKP", "DEF", "MID", "FWD"],
                           title="Last season's total points by position (whole pool)")

    teams_df = load_table("teams").sort_values("strength_attack_home", ascending=False)
    fig2 = go_grouped_bar(teams_df["short_name"].tolist(),
                           {"Home attack": teams_df["strength_attack_home"].tolist(), "Home defence": teams_df["strength_defence_home"].tolist()},
                           title="Home attack/defence strength rating by team")

    hist_df = load_table("historical_gw")
    by_season = hist_df.groupby("season").agg(rows=("element_id", "count"), players=("element_id", "nunique")).reset_index()
    fig3 = go_grouped_bar(by_season["season"].tolist(), {"Player-gameweek rows": by_season["rows"].tolist()},
                           title="Historical training rows per season")

    top = df.sort_values("total_points", ascending=False).head(30)
    table_rows = "".join(
        f"<tr><td>{r.web_name}</td><td>{r.team_name}</td><td>{r.position_name}</td><td>£{r.now_cost/10:.1f}m</td><td>{r.total_points}</td></tr>"
        for r in top.itertuples()
    )

    body = (
        chart_html(fig1)
        + "<h2>Teams</h2>" + chart_html(fig2)
        + "<h2>Historical training data</h2>" + chart_html(fig3)
        + '<div class="note">Players are tracked by the dataset\'s own numeric element_id, not name -- names collide '
          "often enough (two players sharing a name, or literal duplicate source rows) to corrupt training if used as "
          "the identity key. Genuine double-gameweeks are summed into one row per round, matching how FPL actually scores them.</div>"
        + f"<h2>Top 30 players (last season)</h2><table><tr><th>Name</th><th>Team</th><th>Pos</th><th>Price</th><th>Points</th></tr>{table_rows}</table>"
    )
    return page_shell("data_layer.html", "Data Layer", "Everything every model is built on.", body)


# ---------------------------------------------------------------- Model 3

def build_model3() -> tuple[str, list]:
    weights_df = pd.DataFrame(POSITION_WEIGHTS).T
    weights_df.index = ["GK", "DEF", "MID", "FWD"]
    fig_w = go_heatmap(weights_df, title="Weight given to each signal, by position", colorscale="Blues")

    scores = compute_scores()
    scores_df = pd.DataFrame([{"Name": p.web_name, "Position": POSITION_NAMES[p.element_type], "Score": p.score} for p in scores])
    top20 = scores_df.sort_values("Score", ascending=True).tail(20)
    fig_top = go_grouped_bar(top20["Name"].tolist(), {"Score": top20["Score"].tolist()}, horizontal=True)
    fig_top.update_traces(marker_color=[{"GKP": "#E69F00", "DEF": "#0072B2", "MID": "#009E73", "FWD": "#CC79A7"}[p] for p in top20["Position"]])

    fig_dist = go_grouped_violin(scores_df, "Position", "Score", categories=["GKP", "DEF", "MID", "FWD"])

    body = (
        '<div class="note">pts_rate = points per game &middot; value = points per £m &middot; ict = FPL\'s '
        "Influence/Creativity/Threat index &middot; attack = expected goal involvements &middot; defense = -expected "
        "goals conceded. Every score is also multiplied by an availability factor (0 if injured/suspended).</div>"
        + chart_html(fig_w)
        + "<h2>Top 20 (default weights)</h2>" + chart_html(fig_top)
        + "<h2>Score distribution</h2>" + chart_html(fig_dist)
        + "<h2>Final squad</h2>" + squad_section_html(scores)
    )
    return page_shell("model3.html", "Model 3 – Basic Stats", "A transparent, hand-tuned formula -- the baseline the other models are measured against.", body), scores


# ---------------------------------------------------------------- Model 2

def build_model2() -> tuple[str, list]:
    if not M2_MODEL_PATH.exists():
        return page_shell("model2.html", "Model 2 – Statistical", "", '<div class="note">No trained model found.</div>'), []

    bt = get_backtest_arrays()
    model_mae = mean_absolute_error(bt["actual"], bt["predicted"])
    naive_mae = mean_absolute_error(bt["actual"], bt["naive_baseline"])
    improvement = (1 - model_mae / naive_mae) * 100

    sample = bt.sample(min(3000, len(bt)), random_state=1)
    POS_COLORS = {"GK": "#E69F00", "DEF": "#0072B2", "MID": "#009E73", "FWD": "#CC79A7"}
    fig_scatter = go_scatter_by_group(sample, "actual", "predicted", "position", categories=["GK", "DEF", "MID", "FWD"],
                                       colors=POS_COLORS, hover_cols=["name", "round"])
    import plotly.graph_objects as go
    max_val = max(sample["actual"].max(), sample["predicted"].max())
    fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(color="gray", dash="dash"), name="Perfect prediction"))

    by_round = bt.groupby("round").apply(lambda g: pd.Series({
        "Model MAE": mean_absolute_error(g["actual"], g["predicted"]),
        "Naive MAE": mean_absolute_error(g["actual"], g["naive_baseline"]),
    })).reset_index()
    fig_round = go.Figure()
    fig_round.add_trace(go.Scatter(x=by_round["round"], y=by_round["Model MAE"], name="Model", line=dict(color="#0072B2")))
    fig_round.add_trace(go.Scatter(x=by_round["round"], y=by_round["Naive MAE"], name="Naive baseline", line=dict(color="#999999")))
    fig_round.update_layout(xaxis_title="Gameweek", yaxis_title="MAE that gameweek")

    model, X_test, y_test = load_model_and_test_matrix()
    sample_idx = X_test.sample(min(2000, len(X_test)), random_state=1).index
    result = permutation_importance(model, X_test.loc[sample_idx], y_test.loc[sample_idx], n_repeats=5, random_state=1, scoring="neg_mean_absolute_error")
    imp_df = pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean}).sort_values("importance", ascending=True)
    fig_imp = go_grouped_bar(imp_df["feature"].tolist(), {"Importance": imp_df["importance"].tolist()}, horizontal=True)

    scores = predict_next_gameweek()

    body = (
        f'<div class="metrics">{metric("Model MAE", f"{model_mae:.3f} pts")}{metric("Naive baseline MAE", f"{naive_mae:.3f} pts")}{metric("Improvement", f"{improvement:.1f}%")}</div>'
        + "<h2>Predicted vs. actual (held-out 2025-26 season)</h2>" + chart_html(fig_scatter)
        + "<h2>Error over the season</h2>" + chart_html(fig_round)
        + "<h2>Feature importance</h2>" + chart_html(fig_imp)
        + "<h2>Final squad (live prediction for the next gameweek)</h2>" + squad_section_html(scores)
    )
    return page_shell("model2.html", "Model 2 – Statistical Predictor", "Gradient-boosted trees, trained on 3 seasons of real gameweek outcomes.", body), scores


# ---------------------------------------------------------------- Model 1

def build_model1() -> tuple[str, list]:
    if not M1_MODEL_PATH.exists():
        return page_shell("model1.html", "Model 1 – Evolutionary", "", '<div class="note">No trained genome found.</div>'), []

    with open(M1_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    history_df = pd.DataFrame(bundle["history"])
    genome = bundle["genome"]
    evolved, baseline = bundle["test_result"], bundle["baseline_result"]

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["generation"], y=history_df["best"], mode="lines+markers", name="Best genome", line=dict(color="#E69F00")))
    fig.add_trace(go.Scatter(x=pd.concat([history_df["generation"], history_df["generation"][::-1]]),
                              y=pd.concat([history_df["best"], history_df["worst"][::-1]]),
                              fill="toself", fillcolor="rgba(150,150,150,0.15)", line=dict(color="rgba(0,0,0,0)"),
                              name="Population spread", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=history_df["generation"], y=history_df["avg"], mode="lines+markers", name="Population average", line=dict(color="#999999", dash="dot")))
    fig.update_layout(xaxis_title="Generation", yaxis_title="Fitness")

    weights_df = pd.DataFrame({pos: genome.weights[pos] for pos in POSITIONS}).T
    fig_w = go_heatmap(weights_df, title="Evolved genome's weights (final)", colorscale="RdBu", zmid=0)

    compare_df = pd.DataFrame([
        {"Genome": "Evolved", "Total points": evolved["total_points"], "Hits taken": evolved["hits_taken"]},
        {"Genome": "Naive baseline", "Total points": baseline["total_points"], "Hits taken": baseline["hits_taken"]},
    ])
    fig_cmp = go_grouped_bar(compare_df["Genome"].tolist(),
                              {"Total points": compare_df["Total points"].tolist(), "Hits taken": compare_df["Hits taken"].tolist()},
                              colors={"Total points": "#E69F00", "Hits taken": "#999999"})

    weeks_df = pd.DataFrame({"Gameweek": range(1, len(evolved["history"]) + 1), "Evolved": evolved["history"], "Naive baseline": baseline["history"]})
    fig_wk = go.Figure()
    fig_wk.add_trace(go.Scatter(x=weeks_df["Gameweek"], y=weeks_df["Evolved"], name="Evolved", line=dict(color="#E69F00")))
    fig_wk.add_trace(go.Scatter(x=weeks_df["Gameweek"], y=weeks_df["Naive baseline"], name="Naive baseline", line=dict(color="#999999")))
    fig_wk.update_layout(xaxis_title="Gameweek", yaxis_title="Points that week")

    scores = predict_current_squad_scores(genome=genome)
    diff = evolved["total_points"] - baseline["total_points"]

    body = (
        "<h2>Generational learning curve</h2>" + chart_html(fig)
        + f'<div class="metrics">{metric("Fitness, gen 0", f"{history_df["best"].iloc[0]:.0f}")}{metric("Fitness, final gen", f"{history_df["best"].iloc[-1]:.0f}")}{metric("Population std, final gen", f"{history_df["std"].iloc[-1]:.0f}")}</div>'
        + "<h2>What the evolved genome values</h2>" + chart_html(fig_w)
        + f'<div class="metrics">{metric("Transfer threshold", f"{genome.transfer_threshold:.3f}")}{metric("Hit threshold", f"{genome.hit_threshold:.3f}")}</div>'
        + "<h2>Backtest on held-out 2025-26 (out-of-training)</h2>" + chart_html(fig_cmp)
        + f'<div class="metrics">{metric("Evolution\'s margin", f"{diff:+.0f} pts")}</div>'
        + "<h2>Week-by-week points</h2>" + chart_html(fig_wk)
        + "<h2>Final squad</h2>" + squad_section_html(scores)
    )
    return page_shell("model1.html", "Model 1 – Evolutionary Strategy", "A population of strategies plays historical seasons; the best survive and breed.", body), scores


# ---------------------------------------------------------------- Model 4

def build_model4(base_scores: list) -> tuple[str, list]:
    shortlist = build_shortlist(base_scores)
    shortlist_rows = "".join(
        f"<tr><td>{p.web_name}</td><td>{POSITION_NAMES[p.element_type]}</td><td>{p.score:.3f}</td></tr>"
        for p in sorted(shortlist, key=lambda p: -p.score)
    )
    try:
        headlines = sources.fetch_bbc_football_headlines()
        headline_rows = "".join(f"<tr><td>{h['title']}</td><td>{h['summary']}</td></tr>" for h in headlines)
        headlines_html = f"<table><tr><th>Headline</th><th>Summary</th></tr>{headline_rows}</table>"
    except Exception as e:
        headlines_html = f'<div class="note">Couldn\'t fetch headlines: {e}</div>'

    scores = base_scores
    review_html = ""
    try:
        call_ollama("ping", timeout=3)
        ollama_up = True
    except Exception:
        ollama_up = False

    if ollama_up:
        try:
            from fpl_bot.qualitative.synthesizer import run_qualitative_review
            review = run_qualitative_review("ollama", base_scores=base_scores)
            scores = review["scores"]
            notes_rows = "".join(f"<tr><td>{n.web_name}</td><td>{n.adjustment:+.0%}</td><td>{n.rationale}</td></tr>" for n in review["notes"])
            review_html = (
                (f'<div class="note">{review["overall_notes"]}</div>' if review["overall_notes"] else "")
                + (f"<table><tr><th>Name</th><th>Adjustment</th><th>Rationale</th></tr>{notes_rows}</table>" if review["notes"] else '<div class="note">No adjustments this run.</div>')
            )
        except Exception as e:
            review_html = f'<div class="note">Ollama is running but the review failed: {e}</div>'
    else:
        review_html = ('<div class="note">No live review this snapshot -- Ollama wasn\'t running locally when this was generated. '
                        "Scores below fall back to Model 3's. Run <code>ollama serve</code> and regenerate to include a live review.</div>")

    body = (
        f'<div class="note">Shortlist: top scorers per position from Model 3, plus anyone with an official injury/status note -- {len(shortlist)} players.</div>'
        + f"<h2>Shortlisted players</h2><table><tr><th>Name</th><th>Pos</th><th>Model 3 score</th></tr>{shortlist_rows}</table>"
        + "<h2>Recent BBC Sport headlines</h2>" + headlines_html
        + "<h2>Review</h2>" + review_html
        + "<h2>Final squad</h2>" + squad_section_html(scores)
    )
    return page_shell("model4.html", "Model 4 – Qualitative Agent", "Reads news + injury status for a shortlist of players; an LLM reasons about adjustments.", body), scores


# ---------------------------------------------------------------- Ensemble

def build_ensemble(all_scores: dict) -> list:
    weights_df = pd.DataFrame([{"Model": MODEL_LABELS[k], "Weight": w, "key": k} for k, w in MODEL_WEIGHTS.items()])
    fig_w = go_grouped_bar(weights_df["Model"].tolist(), {"Weight": weights_df["Weight"].tolist()})
    fig_w.update_traces(marker_color=[MODEL_COLORS[k] for k in weights_df["key"]])

    ensemble_scores = compute_ensemble_scores(all_scores)
    rank_maps = {k: percentile_ranks(v) for k, v in all_scores.items()}

    top20 = ensemble_scores[:20]
    rows = "<table><tr><th>Name</th><th>Ensemble score</th>" + "".join(f"<th>{MODEL_LABELS[k]}</th>" for k in ["model1", "model2", "model3", "model4"]) + "</tr>"
    for p in top20:
        rows += f"<tr><td>{p.web_name}</td><td>{p.score:.3f}</td>"
        for k in ["model1", "model2", "model3", "model4"]:
            r = rank_maps.get(k, {}).get(p.player_id)
            rows += f"<td>{r:.2f}</td>" if r is not None else "<td>-</td>"
        rows += "</tr>"
    rows += "</table>"

    agreement = model_agreement(all_scores, top_n=40)
    consensus = sorted(((pid, models) for pid, models in agreement.items() if len(models) >= 3), key=lambda x: -len(x[1]))
    name_lookup = {p.player_id: p.web_name for v in all_scores.values() for p in v}
    consensus_rows = "".join(
        f"<tr><td>{name_lookup.get(pid, pid)}</td><td>{len(models)}</td><td>{', '.join(MODEL_LABELS[m] for m in models)}</td></tr>"
        for pid, models in consensus
    )
    consensus_html = f"<table><tr><th>Name</th><th>Models agreeing</th><th>Which</th></tr>{consensus_rows}</table>" if consensus else '<div class="note">No player currently sits in 3+ models\' top 40.</div>'

    body = (
        '<div class="note">Scores are percentile-ranked within each model first (0=worst, 1=best for that model), then combined by weight -- '
        "raw scores aren't comparable across models (different units).</div>"
        + chart_html(fig_w)
        + "<h2>Top 20 -- how each model voted</h2>" + rows
        + "<h2>Where the models agree</h2>" + consensus_html
        + "<h2>Final squad</h2>" + squad_section_html(ensemble_scores)
    )
    page = page_shell("ensemble.html", "Ensemble (Team E)", "How the four models' rankings get combined into one decision.", body)
    return page, ensemble_scores


# ---------------------------------------------------------------- Teams & Approvals

def build_fixture_heatmap(model3_scores: list, horizon: int = 5) -> str:
    from fpl_bot.features.fixtures import fixture_heatmap_grid, player_fixture_features
    import plotly.graph_objects as go

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
            colorbar={"title": "Ease"},
        )
    )
    fig.update_layout(yaxis={"autorange": "reversed"})

    from fpl_bot import db as _db

    conn = _db.get_connection()
    try:
        teams = {r[0]: r[1] for r in conn.execute("SELECT id, short_name FROM teams")}
    finally:
        conn.close()

    fixture_feats = player_fixture_features(horizon)
    defenders = sorted([p for p in model3_scores if p.element_type in (1, 2)], key=lambda p: -p.score)[:20]
    rows = "".join(
        f"<tr><td>{p.web_name}</td><td>{teams.get(p.team_id, '?')}</td>"
        f"<td>{'GKP' if p.element_type == 1 else 'DEF'}</td><td>£{p.now_cost/10:.1f}m</td>"
        f"<td>{p.score:.3f}</td><td>{fixture_feats.get(p.player_id, {}).get('ease', 0):.2f}</td></tr>"
        for p in defenders
    )

    body = (
        '<div class="note">FPL rates every fixture 1-5 for difficulty. Here that is converted to an "ease" score -- '
        "<strong>green = easy, red = hard</strong> -- so a horizontal band of green is a team with a good run. "
        "Blank gameweeks show as a gap (no fixture means guaranteed zero points, which is worse than a hard "
        "fixture, not neutral); double gameweeks count as two scoring chances. Sorted best run first.</div>"
        + chart_html(fig, height=max(420, 26 * len(team_names)))
        + f"<h2>Best defensive picks over the next {horizon} gameweeks</h2>"
        + '<div class="note">Model 3 already weights fixtures most heavily for goalkeepers and defenders, whose '
          "clean-sheet points depend far more on the opponent than attacking returns do.</div>"
        + f"<table><tr><th>Player</th><th>Team</th><th>Pos</th><th>Price</th><th>Model 3 score</th><th>Fixture ease</th></tr>{rows}</table>"
        + '<div class="note"><strong>Note on the word "heatmap":</strong> this is a fixture-difficulty heatmap (the '
          "standard FPL fixture ticker). Player <em>positional</em> heatmaps -- where a player physically touches the "
          "ball -- aren't available from the FPL API and would need a separate source such as Understat or FBref.</div>"
    )
    return page_shell("fixtures.html", "Fixture Heatmap",
                       "Who has the easiest run coming up -- and which defenders that makes attractive.", body)


def build_teams(all_scores_with_ensemble: dict) -> str:
    body = ""
    for key in ["model1", "model2", "model3", "model4", "ensemble"]:
        body += f"<h2>{MODEL_LABELS.get(key, key)}</h2>"
        body += squad_section_html(all_scores_with_ensemble.get(key, []))
    return page_shell("teams.html", "Teams & Squads", "The actual recommended squad for all 5 teams, as of this snapshot.", body)


def build_approvals() -> str:
    pending = list_pending_approvals()
    if pending:
        rows = "".join(f"<tr><td>{MODEL_LABELS.get(p['team_key'], p['team_key'])}</td><td>GW{p['gw']}</td><td>{p['kind']}</td><td>{p['created_at']}</td></tr>" for p in pending)
        body = f"<table><tr><th>Team</th><th>GW</th><th>Kind</th><th>Created</th></tr>{rows}</table>"
    else:
        body = '<div class="note">Nothing pending in the local approval queue right now.</div>'
    body += '<p>The approval queue and mixed-autonomy execution logic run locally -- this page reflects local state as of the last snapshot, not a live feed.</p>'
    return page_shell("approvals.html", "Approval Queue", "Mixed-autonomy rule: captain/bench changes auto-apply, squad changes wait here.", body)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing all models...")
    model_result = compute_all_model_scores(qualitative_backend="ollama")
    scores = {k: v for k, v in model_result["scores"].items() if k != "model4"}  # model4 rebuilt below with headlines context

    print("Building Model 3...")
    model3_html, model3_scores = build_model3()
    scores["model3"] = model3_scores

    print("Building Model 2...")
    model2_html, model2_scores = build_model2()
    scores["model2"] = model2_scores

    print("Building Model 1...")
    model1_html, model1_scores = build_model1()
    scores["model1"] = model1_scores

    print("Building Model 4...")
    model4_html, model4_scores = build_model4(model3_scores)
    scores["model4"] = model4_scores

    print("Building Ensemble...")
    ensemble_html, ensemble_scores = build_ensemble(scores)

    all_with_ensemble = {**scores, "ensemble": ensemble_scores}

    print("Building Home, Data Layer, Teams, Approvals...")
    pages = {
        "index.html": build_home(scores),
        "data_layer.html": build_data_layer(),
        "model3.html": model3_html,
        "model2.html": model2_html,
        "model1.html": model1_html,
        "model4.html": model4_html,
        "ensemble.html": ensemble_html,
        "fixtures.html": build_fixture_heatmap(model3_scores),
        "teams.html": build_teams(all_with_ensemble),
        "approvals.html": build_approvals(),
    }

    for filename, html in pages.items():
        (OUT_DIR / filename).write_text(html, encoding="utf-8")
        print(f"  wrote {filename}")

    print(f"\nDone. {len(pages)} pages written to {OUT_DIR}")


if __name__ == "__main__":
    run()
