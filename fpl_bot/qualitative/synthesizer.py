"""Model 4 -- qualitative synthesis: bundle news/headlines for a shortlist of
players, ask an LLM (local Ollama or cloud Anthropic) to reason about them,
and turn that into a score adjustment layered on top of a base numeric model.

Deliberately a single-pass "read this bundle, answer in JSON" call rather
than a multi-turn tool-calling agent that decides what to fetch next --
simpler, cheaper, and fully sufficient for a daily batch job with a fixed,
curated source list.
"""

import json
from dataclasses import dataclass

from fpl_bot.models.basic_stats import PlayerScore, compute_scores as basic_stats_scores
from fpl_bot.qualitative import sources
from fpl_bot.qualitative.llm_backends import get_backend

SHORTLIST_PER_POSITION = 8
MAX_ADJUSTMENT = 0.3  # cap how much a single qualitative call can move a score, +/-30%


@dataclass
class QualitativeNote:
    player_id: int
    web_name: str
    adjustment: float
    rationale: str


def build_shortlist(base_scores: list[PlayerScore]) -> list[PlayerScore]:
    """Top N per position by base score, plus anyone flagged with news --
    keeps the LLM prompt small instead of reviewing all ~570 players."""
    by_position: dict[int, list[PlayerScore]] = {}
    for p in base_scores:
        by_position.setdefault(p.element_type, []).append(p)

    shortlist_ids = set()
    shortlist = []
    for players in by_position.values():
        for p in sorted(players, key=lambda x: x.score, reverse=True)[:SHORTLIST_PER_POSITION]:
            if p.player_id not in shortlist_ids:
                shortlist_ids.add(p.player_id)
                shortlist.append(p)

    flagged_ids = [p.player_id for p in base_scores if p.player_id not in shortlist_ids]
    news = sources.player_status_notes(flagged_ids)
    for p in base_scores:
        if p.player_id in news and p.player_id not in shortlist_ids:
            shortlist_ids.add(p.player_id)
            shortlist.append(p)

    return shortlist


def build_prompt(shortlist: list[PlayerScore], teams: dict[int, str], headlines: list[dict],
                  news_notes: dict[int, str]) -> str:
    player_lines = []
    for p in shortlist:
        note = news_notes.get(p.player_id, "")
        note_txt = f" | status note: {note}" if note else ""
        player_lines.append(
            f"- id={p.player_id}, {p.web_name} ({teams.get(p.team_id, '?')}), "
            f"£{p.now_cost / 10:.1f}m, model score={p.score:.3f}{note_txt}"
        )

    headline_lines = [f"- {h['title']}: {h['summary']}" for h in headlines]

    return f"""You are an FPL (Fantasy Premier League) analyst. Review the shortlisted \
players below alongside the current football news headlines. For each player, decide \
whether recent news/context should adjust their outlook up or down (e.g. injury doubt, \
new manager favoring/benching them, transfer rumours, poor team form, a favourable \
narrative) versus the neutral case of no adjustment.

SHORTLISTED PLAYERS:
{chr(10).join(player_lines)}

RECENT FOOTBALL HEADLINES:
{chr(10).join(headline_lines) if headline_lines else "(none available)"}

Respond with ONLY valid JSON, no other text, in exactly this shape:
{{
  "players": [
    {{"id": <player id int>, "adjustment": <float from -{MAX_ADJUSTMENT} to {MAX_ADJUSTMENT}>, "rationale": "<one sentence>"}}
  ],
  "notes": "<one or two sentence overall summary of anything notable this week>"
}}

Only include players where you have an actual reason to adjust them (skip players with no \
relevant news -- don't include them with adjustment 0). If nothing in the headlines is \
relevant to any shortlisted player, return an empty "players" list."""


def parse_response(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"players": [], "notes": "(model did not return parseable JSON)"}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"players": [], "notes": "(model returned malformed JSON)"}


def run_qualitative_review(backend: str, base_scores: list[PlayerScore] | None = None) -> dict:
    """Full review: returns {"scores": [...], "notes": [...], "raw_notes": str}."""
    from fpl_bot import db

    base_scores = base_scores or basic_stats_scores()
    shortlist = build_shortlist(base_scores)

    conn = db.get_connection()
    try:
        teams = {row[0]: row[1] for row in conn.execute("SELECT id, short_name FROM teams")}
    finally:
        conn.close()

    headlines = sources.fetch_bbc_football_headlines()
    news_notes = sources.player_status_notes([p.player_id for p in shortlist])

    prompt = build_prompt(shortlist, teams, headlines, news_notes)
    backend_fn = get_backend(backend)
    raw = backend_fn(prompt)
    parsed = parse_response(raw)

    notes = []
    adjustments: dict[int, QualitativeNote] = {}
    for item in parsed.get("players", []):
        try:
            pid = int(item["id"])
            adj = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, float(item["adjustment"])))
        except (KeyError, TypeError, ValueError):
            continue
        note = QualitativeNote(
            player_id=pid,
            web_name=next((p.web_name for p in shortlist if p.player_id == pid), f"#{pid}"),
            adjustment=adj,
            rationale=item.get("rationale", ""),
        )
        adjustments[pid] = note
        notes.append(note)

    adjusted = []
    for p in base_scores:
        if p.player_id in adjustments:
            new_score = round(p.score * (1 + adjustments[p.player_id].adjustment), 4)
            adjusted.append(PlayerScore(p.player_id, p.web_name, p.team_id, p.element_type, p.now_cost, max(new_score, 0)))
        else:
            adjusted.append(p)
    adjusted.sort(key=lambda x: x.score, reverse=True)

    return {
        "scores": adjusted,
        "notes": notes,
        "overall_notes": parsed.get("notes", ""),
    }
