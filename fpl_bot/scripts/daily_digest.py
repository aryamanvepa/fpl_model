"""The Phase 6 daily digest: run all 4 models + the ensemble, diff each
team's recommendation against what it currently holds, auto-apply
captain/bench-only changes, queue anything that changes squad composition
(transfers, or the very first-ever draft) for your approval, and render a
readable summary -- the thing that eventually gets emailed every day.

Run with: python -m fpl_bot.scripts.daily_digest
"""

import io
import sys
import warnings
from datetime import date

warnings.filterwarnings("ignore")

from fpl_bot import db
from fpl_bot.ensemble.combine import compute_ensemble_scores, model_agreement
from fpl_bot.ensemble.models_registry import MODEL_LABELS, compute_all_model_scores
from fpl_bot.ensemble.team_state import diff_squads, get_team_state, list_pending_approvals, queue_approval, save_team_state
from fpl_bot.ingest import ingest_bootstrap
from fpl_bot.optimizer.squad_optimizer import build_squad_result

TEAM_LABELS = {**MODEL_LABELS, "ensemble": "Ensemble (Team E)"}


def _current_gw() -> int:
    conn = db.get_connection()
    try:
        row = conn.execute("SELECT id FROM gameweeks WHERE is_next = 1").fetchone()
        if row is None:
            row = conn.execute("SELECT id FROM gameweeks WHERE finished = 0 ORDER BY id LIMIT 1").fetchone()
        return row[0] if row else 1
    finally:
        conn.close()


def _name_lookup() -> dict[int, str]:
    conn = db.get_connection()
    try:
        return {row[0]: row[1] for row in conn.execute("SELECT id, web_name FROM players")}
    finally:
        conn.close()


def process_team(team_key: str, scores, gw: int) -> dict:
    result = build_squad_result(scores)
    new_squad_ids = [p.player_id for p in result.squad]
    new_starting_ids = [p.player_id for p in result.starting_xi]
    old_state = get_team_state(team_key)

    if old_state is None:
        already_queued = any(
            a["kind"] == "initial_draft" and a["gw"] == gw for a in list_pending_approvals(team_key)
        )
        if not already_queued:
            queue_approval(
                team_key, gw, "initial_draft",
                {"squad": new_squad_ids, "captain": result.captain.player_id if result.captain else None,
                 "cost": result.total_cost},
            )
        return {"status": "pending_initial_draft", "result": result}

    transfers_out, transfers_in = diff_squads(old_state, new_squad_ids)
    if transfers_out or transfers_in:
        already_queued = any(a["kind"] == "transfer" and a["gw"] == gw for a in list_pending_approvals(team_key))
        if not already_queued:
            queue_approval(
                team_key, gw, "transfer",
                {"out": transfers_out, "in": transfers_in,
                 "new_captain": result.captain.player_id if result.captain else None},
            )
        return {"status": "pending_transfer", "result": result, "transfers_out": transfers_out, "transfers_in": transfers_in}

    # squad composition unchanged -- captain/bench-only changes auto-apply, no approval needed
    captain_changed = old_state.captain_id != (result.captain.player_id if result.captain else None)
    bench_changed = old_state.starting_ids != new_starting_ids
    save_team_state(team_key, gw, result)
    return {"status": "auto_applied", "result": result, "captain_changed": captain_changed, "bench_changed": bench_changed}


def render_digest(gw: int, team_reports: dict, errors: dict, model4_notes, model4_overall, agreement_top: list[tuple]) -> str:
    lines = []
    lines.append(f"FPL DAILY DIGEST -- {date.today().isoformat()} -- Gameweek {gw}")
    lines.append("=" * 70)

    if errors:
        lines.append("")
        lines.append("UNAVAILABLE THIS RUN:")
        for key, err in errors.items():
            lines.append(f"  {TEAM_LABELS.get(key, key)}: {err}")

    for key in ["model1", "model2", "model3", "model4", "ensemble"]:
        report = team_reports.get(key)
        lines.append("")
        lines.append(f"-- {TEAM_LABELS[key]} --")
        if report is None:
            lines.append("  (no data this run)")
            continue

        result = report["result"]
        cap = result.captain.web_name if result.captain else "?"
        vc = result.vice_captain.web_name if result.vice_captain else "?"

        if report["status"] == "pending_initial_draft":
            lines.append(f"  NEEDS YOUR APPROVAL: initial squad draft ({len(result.squad)} players, "
                          f"£{result.total_cost / 10:.1f}m). Captain: {cap}, VC: {vc}.")
        elif report["status"] == "pending_transfer":
            names = _name_lookup()
            out_names = ", ".join(names.get(i, str(i)) for i in report["transfers_out"])
            in_names = ", ".join(names.get(i, str(i)) for i in report["transfers_in"])
            lines.append(f"  NEEDS YOUR APPROVAL: transfer OUT [{out_names}] IN [{in_names}]")
            lines.append(f"  (new captain would be {cap}, held until this transfer is approved)")
        else:
            lines.append(f"  Auto-applied. Captain: {cap} ({'changed' if report['captain_changed'] else 'unchanged'}), "
                          f"VC: {vc}. Bench {'reshuffled' if report['bench_changed'] else 'unchanged'}.")

    if model4_overall or model4_notes:
        lines.append("")
        lines.append("-- Model 4 rationale --")
        if model4_overall:
            lines.append(f"  {model4_overall}")
        for note in model4_notes:
            lines.append(f"  {note.web_name}: {note.adjustment:+.0%} -- {note.rationale}")

    if agreement_top:
        lines.append("")
        lines.append("-- Cross-model consensus (players in 3+ models' top 40) --")
        names = _name_lookup()
        for pid, models in agreement_top:
            lines.append(f"  {names.get(pid, pid)}: picked by {', '.join(models)}")

    pending = list_pending_approvals()
    lines.append("")
    lines.append(f"{len(pending)} item(s) awaiting approval -- run `python -m fpl_bot.scripts.approve_pending` to review.")
    lines.append("=" * 70)
    return "\n".join(lines)


def run(refresh: bool = True, qualitative_backend: str = "ollama") -> str:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if refresh:
        ingest_bootstrap.run()

    gw = _current_gw()
    model_result = compute_all_model_scores(qualitative_backend=qualitative_backend)
    scores = model_result["scores"]

    team_reports = {}
    for key in ["model1", "model2", "model3", "model4"]:
        if key in scores:
            team_reports[key] = process_team(key, scores[key], gw)

    if scores:
        ensemble_scores = compute_ensemble_scores(scores)
        team_reports["ensemble"] = process_team("ensemble", ensemble_scores, gw)

    agreement = model_agreement(scores) if scores else {}
    agreement_top = sorted(
        ((pid, models) for pid, models in agreement.items() if len(models) >= 3),
        key=lambda x: -len(x[1]),
    )[:15]

    digest = render_digest(gw, team_reports, model_result["errors"], model_result["model4_notes"],
                            model_result["model4_overall"], agreement_top)
    print(digest)
    return digest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama", choices=["ollama", "anthropic"],
                         help="LLM backend for Model 4's qualitative review")
    parser.add_argument("--email", action="store_true",
                         help="also email the digest (requires SMTP_HOST/SMTP_USER/SMTP_PASSWORD env vars)")
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()

    digest_text = run(refresh=not args.no_refresh, qualitative_backend=args.backend)

    if args.email:
        from fpl_bot.notify.email import send_digest_email

        send_digest_email(f"FPL Daily Digest -- {date.today().isoformat()}", digest_text)
        print("\nEmailed digest.")
