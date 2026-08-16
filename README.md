# FPL Multi-Model Bot

A system that manages Fantasy Premier League squads using four genuinely different
decision-making approaches, run against each other as five live teams, combined by
a weighted ensemble, and surfaced through a dashboard that shows the actual internals
of every model, not just the final answer.

Built with [Claude Code](https://claude.com/claude-code).

## The four models

| | Approach | Real, measured result |
|---|---|---|
| **Model 1** | Evolutionary strategy (genetic algorithm) -- a population of transfer/captaincy policies plays simulated historical seasons; the best-scoring survive and breed the next generation | Beat a sensible non-evolved baseline by **+101 points** over a held-out season it never trained on, while taking a third as many transfer hits |
| **Model 2** | Statistical predictor -- gradient-boosted trees trained on 3 seasons of real gameweek outcomes, predicting expected points from pre-gameweek-known features only | **MAE 1.02 vs 1.08** for a naive rolling-average baseline on a held-out season (~5% improvement) |
| **Model 3** | Basic stats -- a transparent, hand-tuned formula (points-per-game, value, ICT index, xG involvement, defensive xG conceded), weighted per position | The fast-to-ship baseline the other models get measured against |
| **Model 4** | Qualitative agent -- an LLM (local Ollama or cloud Anthropic API) reads recent football news and official injury/status notes for a shortlist of players and reasons about score adjustments | Single-pass synthesis, not a multi-turn tool-calling agent -- deliberately simple for a fixed daily source list |

Each model's output is a per-player score, fed into a shared **ILP optimizer** (PuLP)
that picks the actual best 15-man squad under real FPL constraints (budget, 3-per-club
max, valid formation) -- so every model produces a genuinely valid, legal squad, not
just a ranking.

## Architecture

```
Data layer (official FPL API + historical gameweek data)
        |
Model 1        Model 2        Model 3        Model 4
(evolutionary)  (statistical)  (basic stats)  (qualitative)
        \___________|_______________|______________/
                          |
                Ensemble (rank-normalized weighted average)
                          |
        5 live teams, one mixed-autonomy ruleset:
        captain/bench changes auto-apply, transfers queue for approval
                          |
        Weekly scheduler (deadline-aware, idempotent per gameweek)
                          |
                    Streamlit dashboard
```

The ensemble doesn't average raw scores across models -- they're in incompatible
units (predicted points vs. a genome-weighted linear score vs. a 0-1 composite).
Each model's scores are converted to a **percentile rank within that model** first,
then combined with weights that reflect what's actually been backtested (Models 1
and 2 get the most weight; Model 4 the least, since it hasn't had a live LLM
backtest yet).

## Dashboard

A multi-page Streamlit app (`fpl_bot/dashboard/`) exposing the actual internals of
every model, not just outputs:

- **Data Layer** -- the raw players/teams/fixtures/historical data everything is built on
- **Model 3** -- the scoring formula as live, draggable weight sliders; a signal-vs-actual-outcome correlation chart
- **Model 2** -- backtest accuracy, a predicted-vs-actual scatter, on-demand permutation feature importance, error drift over a season
- **Model 1** -- the generation-by-generation fitness curve (the actual evolutionary learning curve), a slider to inspect the evolved genome's weights at any point in training, and what squad that generation's strategy would draft *today*
- **Model 4** -- the live shortlist and news it reads, with a button to trigger a real LLM call
- **Ensemble** -- live-adjustable model weights, and a full per-model rank-contribution table showing exactly how each player's ranking was arrived at
- **Teams & Squads** / **Approval Queue** -- all 5 teams' actual current squads, and a working approve/reject flow

```bash
pip install -r requirements.txt
streamlit run fpl_bot/dashboard/Home.py
```

## Automation

`fpl_bot/scripts/scheduled_run.py`, registered as a Windows Scheduled Task, checks
daily whether the next gameweek deadline is within 60 hours (rather than a blind
fixed weekly day, which would drift whenever a deadline shifts) and, once per
gameweek, refreshes data, runs all 4 models + ensemble, diffs against each team's
held squad, auto-applies captain/bench changes, and queues transfers for approval.

```bash
python -m fpl_bot.scripts.daily_digest        # run once manually
python -m fpl_bot.scripts.approve_pending      # review/act on the queue
```

## Known limitations

This is a working system, not a finished product -- a few things are deliberate v1
simplifications rather than oversights:

- Model 1's simulator: at most one transfer evaluated per week, free-transfer bank
  caps at 2 (not the newer 5-stack rule), no bench autosubs, no chips
- Model 4 hasn't had a live LLM backtest in this environment (no Ollama/API key
  configured during development) -- the pipeline is fully tested with a stubbed
  response, but real-world accuracy is unmeasured
- No real FPL account integration yet -- the 5 teams' state is tracked locally,
  not against actual FPL accounts

## Data sources

- The official (public, read-only) [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/)
- [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) --
  historical gameweek-level data used to train and backtest Models 1 and 2
- BBC Sport's public football RSS feed, for Model 4
