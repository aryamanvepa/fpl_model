# FPL Multi-Model Bot System — Plan

Stack: Python. Autonomy: mixed (auto-execute low-risk moves like captain/bench order; require approval for transfers and chips). Starting point: brand-new team, 2026/27 season about to start, so we need an initial squad draft, not just weekly transfer suggestions.

## 0. One terminology fix

The "dino game" model isn't unsupervised learning — the Chrome dino AI you're thinking of is trained with **NEAT (neuroevolution) or a genetic algorithm**: a population of candidate strategies plays, the best-scoring ones survive and breed/mutate into the next generation. That maps really well onto FPL (a "genome" = a squad/transfer policy, "fitness" = points scored across a simulated season), so Model 1 below is built as an evolutionary strategy model, not unsupervised clustering.

## 1. Architecture (5 layers)

```
Data Layer → 4 Model Layer → Ensemble/Optimizer → Execution Layer → Interface
```

Each model outputs the same shape: a per-player score/predicted-points for the upcoming gameweek(s), plus (for Models 1 and 4) higher-level suggestions like chip timing. The Ensemble layer reconciles them into one squad decision.

## 2. Data layer

- **Official FPL API** (free, public, no auth needed for reads): `bootstrap-static` (players/teams/prices), `fixtures`, `element-summary/{id}` (per-player history), `entry/{id}` (your team). This is the backbone — safe and stable.
- **Historical data**: the `vaastav/Fantasy-Premier-League` GitHub repo has cleaned gameweek-by-gameweek data back to ~2016 — essential for backtesting Models 1–3 before risking real transfers.
- **Advanced stats (xG/xA)**: Understat, FBref. Scraping is technically possible but check robots.txt/ToS per site; rate-limit and cache aggressively so we hit them rarely.
- **Qualitative/news**: injury news, press conferences, and FPL analysis sites (e.g. Fantasy Football Scout is subscription-walled — flag this, we'll need to pick sources that are actually scrapable, like free previews, r/FantasyPL, official club news, and press-conference transcripts).
- **Storage**: SQLite to start (zero setup), Postgres later if it grows. One ETL script pulls fresh data on a schedule.

## 3. Model 1 — Evolutionary strategy model

- Genome = a strategy (weights over the featuers/signals used to decide transfers, captaincy, chip timing).
- Fitness = total points scored when that genome's policy is replayed against historical seasons.
- Each "epoch" = one simulated season; selection + crossover + mutation produce the next generation of strategies.
- Library: start with a simple custom GA, graduate to `DEAP` or `neat-python` if useful.
- This is the most experimental and slowest to pay off — sequence it last.

## 4. Model 2 — Statistical player-performance predictor

- Predicts expected points per player per gameweek.
- Features: recent form, xG/xA, minutes-played trend, fixture difficulty, home/away, opponent defensive strength, set-piece role, price momentum.
- Models: gradient-boosted trees (XGBoost/LightGBM) for expected points; Poisson/negative-binomial regression for goals/assists specifically.
- Validate by backtesting predictions against actual historical gameweek scores.

## 5. Model 3 — Basic statistical analysis

- Transparent, explainable scoring: points-per-million (value), form, fixture run (next 3-5 games), ownership/differential angle, price-change momentum.
- No ML — just a scoring formula. Cheap to build, useful as a baseline/sanity check against Models 1 and 2, and as the fastest thing we can ship before the season deadline.

## 6. Model 4 — Qualitative AI agent

- An LLM agent with web-search/scraping tools that reads a curated, ToS-respecting source list (injury news, press conferences, free analysis, community sentiment) and produces a written rationale: consensus transfer targets, differential picks, injury risk flags.
- Output feeds the ensemble as an adjustment layer (e.g. "downweight player X, likely rotated" or "market is moving on player Y").

## 7. Ensemble / optimizer

- Combine the four models' outputs into one predicted-points table, weighted by each model's backtested historical accuracy.
- Feed that into an **integer linear program** (via `PuLP` or `python-mip`) that picks the optimal 15-man squad / XI / captain under FPL's actual constraints (budget, 3-per-club max, formation rules). This solver is the same for pre-season draft and weekly transfers — only the objective inputs change.

## 8. Execution layer — 5 live teams, one ruleset

Each model gets its own real FPL team so we can compare live performance, not just backtests, plus one more team that the ensemble manages:

| Team | Managed by |
|---|---|
| Team A | Model 1 (evolutionary strategy) |
| Team B | Model 2 (statistical predictor) |
| Team C | Model 3 (basic stats) |
| Team D | Model 4 (qualitative agent) |
| Team E ("main") | Ensemble of all four |

All five follow the **same mixed-autonomy rule**:
- **Auto-executed**: captain/vice-captain pick, bench order — low risk, easily correctable next week.
- **Approval required**: transfers (cost points if unplanned/extra), chip usage (Wildcard, Free Hit, Bench Boost, Triple Captain) — bot proposes, notifies you, you confirm before it executes.

You'll create and hand over 5 sets of FPL login credentials later (Phase 5) — the system should be written against a generic "FPL account" abstraction from the start so plugging in 5 accounts is just config, not a rewrite.

Note: FPL has no official *write* API — team changes require an authenticated session (login cookies), which is unofficial/fragile and can break if the site changes. We'll build this carefully and keep manual fallback easy, across all 5 accounts.

## 8a. Daily digest & notifications

A scheduled job runs **once a day** (independent of the weekly deadline cadence, so you get ongoing visibility — price changes, injury news, form shifts — not just deadline-day decisions) and sends a digest containing:

- Each of the 4 models' current recommendations for its own team (transfers considered, captain pick, any chip flagged).
- What the ensemble (Team E) decided or attempted to do, and why (which models it weighted and how).
- Anything sitting in the "needs your approval" queue across all 5 teams, with a simple way to approve/reject.

**Channel plan**: email first (simple SMTP or a transactional email API — e.g. Resend/SendGrid), since it needs no extra account setup. WhatsApp comes later once you've set up a Twilio (or WhatsApp Business API) account — WhatsApp has no free send API, so that's a prerequisite to add the channel, not a code limitation.

## 9. Pre-season: initial squad draft

Since the season hasn't started, Phase 1 needs a standalone "draft my 15" run: apply Model 3 (and Model 2 once ready) + the ILP optimizer to pick a starting squad within the 100m budget before the first deadline.

## 10. Backtesting framework

Replay every model, and the ensemble, gameweek-by-gameweek across historical seasons from the vaastav dataset. Compare against actual outcomes and the FPL average-manager score — this is how we decide model weights in step 7 and catch a bad model before it touches real transfers.

## 11. Infrastructure & scheduling

Weekly cadence tied to FPL deadlines (typically Friday evening/Saturday before each gameweek): scheduled job pulls fresh data → runs models → ensemble → auto-executes low-risk items → sends approval request for the rest. Local Python + SQLite + a cron/scheduled task is enough to start.

## 12. Suggested build order

1. **Phase 0** — Data pipeline: FPL API client + historical data ingestion into SQLite.
2. **Phase 1** — Model 3 (basic stats) + ILP optimizer + initial squad draft. Priority: get a working baseline squad picked before the season deadline.
3. **Phase 2** — Model 2 (statistical predictor), backtested against historical seasons.
4. **Phase 3** — Model 4 (qualitative agent) with a vetted, scrapable source list.
5. **Phase 4** — Model 1 (evolutionary strategy) — most experimental, build once the others give it something to beat.
6. **Phase 5** — Ensemble layer (Team E) + mixed-autonomy execution/approval flow, generic "FPL account" abstraction ready for 5 credentials.
7. **Phase 6** — Daily email digest (all 4 models + ensemble + pending approvals).
8. **Phase 7** — Wire up the 5 real accounts once you provide credentials; add WhatsApp channel once Twilio/WhatsApp Business is set up.

## Open items to settle before/at each phase

- Exact qualitative source list for Model 4 (need scrapable, ToS-compliant sites — Fantasy Football Scout's paid content is off-limits).
- The 5 FPL account credentials (Teams A-E) — you'll create and hand these over later; Phase 5 builds against a generic account abstraction so this plugs in without a rewrite.
- Email provider choice for the digest (SMTP vs. an API like Resend/SendGrid) — pick at Phase 6.
- Twilio/WhatsApp Business setup — needed before Phase 7's WhatsApp channel; email works without it.
- Auth approach for the unofficial FPL write API — pick when we reach Phase 5.

## Status

**Phase 0/1 built** in `fpl_bot/`: FPL API client, SQLite ingest, Model 3 basic-stats scoring, ILP squad optimizer, end-to-end draft script.

```bash
pip install -r requirements.txt
python -m fpl_bot.scripts.draft_squad
```

Pulls live data, scores every player, and prints the optimal 15-man squad, valid starting XI, captain/vice-captain, bench order, and total cost under the real £100m / 3-per-club / formation constraints. Re-running re-pulls fresh data each time — this same script is the seed for the Phase 6 daily-digest job later.

**Phase 2 built**: Model 2, the statistical predictor.

- Trained on 3 seasons of historical gameweek data (2023-24, 2024-25, 2025-26) pulled from the public `vaastav/Fantasy-Premier-League` dataset — ~87k player-gameweek rows.
- Features are all "pre-gameweek known" (no leakage): rolling 3-game points/minutes, team and opponent rolling goals-for/against rates, home/away, price, position.
- Model: `HistGradientBoostingRegressor` (scikit-learn). Backtested on the held-out 2025-26 season: **MAE 1.00 points/player/gameweek vs. 1.06 for a naive rolling-average baseline** (~5% improvement) — a real but modest edge, which is expected given how much of FPL scoring is inherently noisy (injuries, red cards, rotation).
- For the current pre-season pool (no in-season history yet), cold-start inputs use last season's (2025-26) rates as the prior, matched to the actual GW1 fixture list already in the database.

```bash
python -m fpl_bot.models.statistical_predictor   # retrain + backtest, saves fpl_bot/data/model2_predictor.pkl
python -m fpl_bot.scripts.draft_squad_model2      # Model 2's own squad for the next gameweek
```

**Phase 3 built**: Model 4, the qualitative agent, in `fpl_bot/qualitative/`.

- Researched existing FPL/news/Reddit MCP servers on GitHub before building. Verdict: the FPL-specific ones (nguyenanhducs/fpl-mcp-server, rishijatia/fantasy-pl-mcp, etc.) mostly just wrap the same public FPL API we already call directly in `api_client.py`, so they don't add anything new. The fetch/RSS/Reddit MCP servers are themselves thin wrappers over plain HTTP endpoints (fetch a URL, hit Reddit's public JSON). Since this runs as a headless daily batch job rather than an interactive MCP-connected chat session, I called those same underlying endpoints directly with `requests` instead of adding MCP subprocess/JSON-RPC machinery — same data, far fewer moving parts. If you later want this running *as* a live MCP-connected agent inside Claude Code, that's a natural extension of the same source layer.
- **Sources (v1)**: BBC Sport football RSS feed (public, built for this) + each player's official FPL injury/status text (already in our DB, no extra fetch). **Dropped for now**: r/FantasyPL — its public `.json` endpoint 403's without a registered Reddit OAuth app, which is a credential only you can create (same pattern as the FPL accounts); revisit if you want it.
- **Two interchangeable LLM backends**, both plain `requests` calls (no SDK dependency): a local **Ollama** model (free, private, needs Ollama installed + running) and the **cloud Anthropic API** (needs `ANTHROPIC_API_KEY`). Neither was available in this sandbox, so the full pipeline (shortlist building, prompt construction, JSON parsing, score merging, optimizer integration) was verified end-to-end with a stubbed LLM response — logic confirmed correct, but no live LLM call has run yet. Once you have Ollama running locally or export an API key, `python -m fpl_bot.scripts.draft_squad_model4 --backend ollama` (or `--backend anthropic`) will do a real run.
- Design: single-pass "read this bundle, answer in JSON" call, not a multi-turn tool-calling agent that decides what to fetch next — deliberately simpler for a fixed daily batch job. Only a shortlist (~32 players: top scorers per position + anyone with an injury/status note) gets reviewed, not all ~570 players, to keep it fast and cheap regardless of backend.
- Output is a score *adjustment* (±30% max) layered on top of Model 3's base scores, plus human-readable rationale per adjusted player — this rationale text is exactly what the Phase 6 daily digest will surface.

```bash
python -m fpl_bot.scripts.draft_squad_model4 --backend ollama
python -m fpl_bot.scripts.draft_squad_model4 --backend anthropic
```

**Phase 4 built**: Model 1, the evolutionary strategy, in `fpl_bot/models/evolutionary/`.

- **Genome** = per-position weights over 8 signals (recent form, minutes, price, team/opponent scoring rates, home/away) plus two policy thresholds: how big a score gain has to be to spend a free transfer, and how big to justify a -4 hit.
- **Simulator (the "environment")**: replays one real historical season gameweek-by-gameweek — draft an initial squad, then each week decide transfers/captaincy from only pre-gameweek-known information, apply the real historical points that actually happened, track hit penalties. Total season points minus hits = fitness, exactly the score a dino-game agent gets for how far it ran.
- **GA loop**: 30 genomes x 20 generations, tournament selection + crossover + elitism + mutation, trained on 2023-24 + 2024-25. Fitness climbed generation over generation (2695 → 4114 best, 1395 → 3763 average across the population) — a real, visible learning curve, not just noise.
- **Backtest on held-out 2025-26**: the evolved genome scored **1557 pts vs. 1473 for a sensible non-evolved baseline** (a genome that just trusts recent form) — both run through the identical simulator for a fair comparison. Notably, evolution took a third as many hits (12 vs. 34) for almost the same raw points, i.e. it learned transfer *discipline*, not just better player-picking.
- **Known v1 simplifications**, worth revisiting later: at most one transfer evaluated per week (not two), free-transfer bank caps at 2 (not the newer 5-stack rule), no bench autosubs, no chips (wildcard/free hit/bench boost/triple captain).
- **Bug fixed along the way**: the historical dataset had ~1,070 duplicate-ish rows and used player *name* as an identity key, which breaks when two players share a name or a row is a literal duplicate. Switched to the dataset's own numeric player id (`element_id`) and properly aggregated genuine double-gameweeks (summed, not duplicated). This also slightly corrected Model 2's training data — retrained, backtest MAE moved from 1.00 to 1.02 (still ~5% better than naive), a small honest change from fixing real data hygiene, not a regression in the model itself.

```bash
python -m fpl_bot.scripts.train_model1        # evolve + backtest, saves fpl_bot/data/model1_genome.pkl
python -m fpl_bot.scripts.draft_squad_model1  # Model 1's own squad for the next gameweek
```

**Phase 5 + 6 built**: the ensemble layer (Team E), mixed-autonomy state tracking, the approval queue, and the daily digest, in `fpl_bot/ensemble/` and `fpl_bot/notify/`.

- **Ensemble (`ensemble/combine.py`)**: the four models produce scores in incompatible units (Model 2 = predicted points ~0-8, Model 1 = a genome-weighted linear score ~-3 to 3, Models 3/4 = a 0-1 composite), so averaging them directly would just let whichever model happens to use the widest numeric range dominate for no real reason. Fixed by converting each model's scores to a **percentile rank within that model** first, then combining ranks with weights: `model1: 0.30, model2: 0.35, model3: 0.20, model4: 0.15`. The two models with a measured backtest edge get the most weight (roughly balanced against each other); Model 3 (unvalidated hand-tuned baseline) gets less; Model 4 (adjustment-only, no live LLM run yet) gets the least. Weights are a plain dict, easy to revise once more backtesting exists. If Model 4's backend is unavailable that day, it's dropped and the other weights redistribute automatically rather than the whole run failing.
- **Team state (`ensemble/team_state.py`)**: since real FPL account credentials don't exist yet (Phase 7), the system now tracks "what each of the 5 teams currently holds" itself, in SQLite -- this is what lets day-to-day runs propose actual transfers instead of re-drafting from nothing every time. Once real accounts are wired in, this table gets reconciled against the live entry API instead of being the source of truth.
- **Mixed-autonomy approval queue**: captain and bench-order changes auto-apply immediately (no approval needed, matching the rule from the plan). Anything that changes squad *composition* -- the first-ever draft, or a transfer -- gets queued in `pending_approvals` instead of applied. `python -m fpl_bot.scripts.approve_pending` lists the queue and lets you approve/reject (interactively, by id, or `--approve-all`); approving re-runs that model fresh and commits the result as the team's new held state. Re-running the digest before you've approved something does **not** spam duplicate queue entries -- verified this explicitly.
- **Daily digest (`scripts/daily_digest.py`)**: runs all 4 models + the ensemble, diffs each against its held state, auto-applies or queues as above, and renders a plain-text report: what's pending approval per team, Model 4's rationale text, and a **cross-model consensus section** (players ranked in 3+ models' individual top-40 -- e.g. Saka, Semenyo, Rice, Enzo all showed up across Models 1-3 in the first real run). Tested the full loop live: first run queued 4 initial drafts (Model 4 correctly reported unavailable, not silently skipped), approved one, second run correctly recognized that team as stable and auto-applied its captain/bench instead of re-queuing.
- **Email (`notify/email.py`)**: plain SMTP, configured entirely via environment variables (`SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `DIGEST_TO_EMAIL`). `--email` flag on the digest script sends it. Not configured or tested live in this environment (no credentials set) -- verified it fails with a clear, specific error rather than silently no-op'ing or (worse) me sending a real email without asking first. WhatsApp is still the documented Phase 7 follow-up once you set up Twilio/WhatsApp Business.

```bash
python -m fpl_bot.scripts.daily_digest                    # run + print
python -m fpl_bot.scripts.daily_digest --backend anthropic --email  # + cloud LLM + send
python -m fpl_bot.scripts.approve_pending --list           # see what's queued
python -m fpl_bot.scripts.approve_pending                  # interactive approve/reject
```

**Phase 8 built (added after the original plan)**: a comprehensive Streamlit dashboard in `fpl_bot/dashboard/`, covering every model's inner workings -- requested separately, not in the original scope, but documented here for continuity.

```bash
streamlit run fpl_bot/dashboard/Home.py
```

8 pages: **Home** (system status, all 5 teams at a glance), **Data Layer** (players/teams/fixtures browser + historical data quality notes), **Model 3** (the scoring formula as a heatmap, live rankings), **Model 2** (backtest MAE, predicted-vs-actual scatter, on-demand permutation feature importance), **Model 1** (the generation-by-generation fitness curve -- the actual "learning curve" -- the evolved genome's weights, and week-by-week out-of-training performance), **Model 4** (live shortlist/headlines viewer with an on-demand real LLM call button), **Ensemble** (weight rationale, per-model rank contribution per player, consensus table), **Teams & Squads** (all 5 live squads), and **Approval Queue** (working approve/reject buttons wired to the same logic as the CLI).

Two real environment bugs surfaced and fixed while building this (both pre-existing, not caused by this project): this machine's `xarray` build is incompatible with numpy 2.x, which broke `plotly.express` entirely (it eagerly imports xarray at package-init) -- fixed by building all charts on `plotly.graph_objects` instead, which has no such dependency, rather than chasing further package upgrades on a visibly fragile shared Anaconda environment. Separately, this machine's `matplotlib` doesn't import at all (same numpy-2.x ABI family of issue), which broke pandas' `.style.background_gradient()` -- replaced with Streamlit's native `column_config.ProgressColumn`, which needs no matplotlib. Verified every page live in a browser after both fixes, not just import-checked.

**Dashboard round 2 (interactivity + depth), per feedback that the first pass was too static:**
- Every model page now has **live interactive controls**, not just static charts: Model 3's 20 signal weights are draggable sliders that recompute rankings in real time; the Ensemble's 4 model weights are sliders too (with a guard against all-zero weights); Model 2's backtest has position + gameweek-range filters; Model 1 has a **generation slider** that lets you scrub through training and see exactly what the best genome valued at that point -- and, since `predict_current_squad_scores()` already accepted an optional genome, what squad that generation's strategy would draft for the *current* live gameweek.
- Model 1's training now also captures **per-generation genome snapshots and population spread** (best/avg/worst/std), not just best/avg -- required retraining (~5 min) since the old saved model didn't have this data. Caught and fixed a real bug in the process: `statistics.pstdev` chokes on numpy scalar types, so fitness values are now cast to plain Python floats.
- **Every model page (1-4) and the Ensemble page now ends with that model's actual "Final Squad"** (starting XI, bench, captain/VC, cost) computed from whatever's currently on screen -- including the live-adjusted weights/generation/backend -- via a shared `render_squad_section()` component, so you see each model's real output without leaving the page.
- Added `explainer()` -- collapsed-by-default "how to read this" notes under the less-obvious charts (why show population spread, why pts_rate's correlation looks tautologically high, etc.) -- and a Model 3 signal-vs-actual-outcome correlation chart, and a Model 2 error-by-gameweek chart (does accuracy drift over a season?).
- `basic_stats.compute_scores()` now accepts an optional `weights_override` so the dashboard's live sliders don't duplicate the scoring logic.

**Phase 9 built (scheduling, added after the original plan)**: the missing piece that actually makes "weekly" real. Everything up to this point (`daily_digest.py`) only ran when someone typed the command by hand -- there was no automation.

- **`fpl_bot/scripts/scheduled_run.py`**: the automation entry point. Rather than a blind fixed weekly day (which would drift out of sync whenever a deadline shifts -- postponements, double/blank gameweeks, international breaks moving a deadline to Monday), it runs a cheap check *daily* and only does real work (full data refresh, all 4 models + ensemble, diff-and-queue transfers) once the next gameweek's actual deadline is within 60 hours -- adapts automatically to whatever the real deadline is, straight from the `gameweeks` table.
- **Idempotent per gameweek**: a new `digest_runs` table records which gameweek's already had its full run, so being "within 60 hours" on both Thursday and Friday doesn't trigger two runs or duplicate the approval queue. Verified this directly: force-triggered a run, confirmed it queued once, ran again immediately, confirmed it correctly skipped as already-done.
- **Everything logs to `fpl_bot/data/scheduler.log`** (timestamped, append-only) -- since this now runs unattended with nobody watching the console, that log is the only way to see what happened, including full tracebacks if a step ever fails.
- **Registered as a real Windows Scheduled Task** (`FPLBotWeeklyDigest`, daily 08:00, via `run_scheduler.bat` so the working directory and full Python path are always correct) -- runs independently of any Claude Code session, survives this chat ending or the machine rebooting. Triggered it for real through `schtasks /Run` (not just calling the Python script directly) to confirm the whole OS-level path works: exit code 0, correct skip logic, logged correctly.
- Email still only fires if `SMTP_HOST`/`SMTP_USER`/`SMTP_PASSWORD` are set; otherwise it logs "not configured" and leaves the digest sitting in the approval queue for you to check via the dashboard or `approve_pending` CLI -- consistent with earlier: nothing sends on your behalf without you setting the credentials.

To change the time or inspect/remove the task: `schtasks /Query /TN "FPLBotWeeklyDigest" /V /FO LIST`, `schtasks /Change /TN "FPLBotWeeklyDigest" /ST HH:MM`, or `schtasks /Delete /TN "FPLBotWeeklyDigest" /F`.

## What's left

Only **Phase 7** from the original roadmap: wiring in your 5 real FPL account credentials (replacing the local `team_state` table with the real entry API as source of truth), and setting up Twilio/WhatsApp Business for the second notification channel. Everything else in the original plan, plus the dashboard and scheduling, is built and tested.
