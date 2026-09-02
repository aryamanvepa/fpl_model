import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from fpl_bot import db
from fpl_bot.dashboard.utils import MODEL_LABELS, explainer, page_header
from fpl_bot.ensemble.combine import compute_ensemble_scores
from fpl_bot.ensemble.models_registry import compute_all_model_scores
from fpl_bot.ensemble.transfer_suggestions import best_single_swaps, load_players_meta, resolve_names_to_ids
from fpl_bot.vision.screenshot_parser import parse_team_screenshot

st.set_page_config(page_title="Update My Team", page_icon="📸", layout="wide")
page_header("Update My Team", "Upload a screenshot of your real FPL squad and get transfer suggestions from each model.")

explainer(
    """
Only one step here calls a paid API: parsing the screenshot into player names, using Claude
**Haiku** (the cheapest available model) with the image downscaled first and a tight prompt
capped at 400 output tokens -- one small call, once, per screenshot you upload. Matching those
names to real players, scoring them, and ranking transfer suggestions all happen locally in
Python afterward -- no further API calls, regardless of how many suggestions you view or how
often you revisit this page.
""",
    label="Where does this cost money?",
)

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.expander("API key (only needed if ANTHROPIC_API_KEY isn't already set in your environment)"):
    st.session_state.api_key = st.text_input("Anthropic API key", value=st.session_state.api_key, type="password")

uploaded = st.file_uploader("Squad screenshot", type=["png", "jpg", "jpeg"])
if uploaded:
    st.image(uploaded, caption="Uploaded screenshot", width=400)

if uploaded and st.button("Parse screenshot (1 API call)"):
    with st.spinner("Calling Claude Haiku..."):
        try:
            parsed = parse_team_screenshot(uploaded.getvalue(), api_key=st.session_state.api_key or None)
            st.session_state.parsed_squad = parsed
            st.session_state.parse_error = None
        except Exception as e:
            st.session_state.parsed_squad = None
            st.session_state.parse_error = str(e)

if st.session_state.get("parse_error"):
    st.error(st.session_state.parse_error)

parsed = st.session_state.get("parsed_squad")
if parsed:
    st.success("Parsed. Review below -- fix anything misread before saving.")
    all_names = parsed["starting"] + parsed["bench"]
    players_meta = load_players_meta()

    if len(all_names) < 15:
        missing_count = 15 - len(all_names)
        st.warning(
            f"Only found {len(all_names)} players (expected 15) -- the bench goalkeeper is the "
            f"easiest one for this to miss. Type the {missing_count} missing player name(s) below."
        )
        all_names = all_names + [f"(missing player {i + 1})" for i in range(missing_count)]

    # Tracked by slot (position in the 15-name list), not by name string --
    # keying by name breaks if any two of the 15 names collide (e.g. two
    # players sharing a short name), which silently undercounts and leaves
    # "ready" stuck false with no visible reason.
    slot_ids: list[int | None] = [None] * len(all_names)
    for i, name in enumerate(all_names):
        matched, _ = resolve_names_to_ids([name], players_meta)
        if name in matched:
            slot_ids[i] = matched[name]

    matched_count = sum(1 for v in slot_ids if v is not None)
    st.write(f"**{matched_count}/{len(all_names)}** names matched automatically.")
    if matched_count:
        st.dataframe(
            [{"Name (as read)": n, "Matched to": players_meta[pid]["web_name"], "Price": f"£{players_meta[pid]['now_cost']/10:.1f}m"}
             for n, pid in zip(all_names, slot_ids) if pid is not None],
            use_container_width=True, hide_index=True,
        )

    unresolved_slots = [i for i, v in enumerate(slot_ids) if v is None]
    if unresolved_slots:
        st.warning(f"Couldn't confidently match {len(unresolved_slots)} name(s). Type the correct FPL name for each:")
        for i in unresolved_slots:
            fix = st.text_input(f"Correct name for '{all_names[i]}'", key=f"fix_slot_{i}").strip()
            if fix:
                fixed, _ = resolve_names_to_ids([fix], players_meta)
                if fix in fixed:
                    slot_ids[i] = fixed[fix]
                else:
                    st.caption(f"⚠️ '{fix}' didn't match any current player either -- check the spelling.")

    still_unresolved = [all_names[i] for i, v in enumerate(slot_ids) if v is None]
    all_names_resolved = len(still_unresolved) == 0

    # A valid FPL squad is always exactly 2 GK / 5 DEF / 5 MID / 3 FWD -- worth checking
    # explicitly, since a wrong (but confident) name match would otherwise silently
    # produce an invalid mix with no visible symptom until the optimizer breaks on it.
    EXPECTED_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
    POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    composition_ok = True
    if all_names_resolved:
        actual_counts = {pos: 0 for pos in EXPECTED_COUNTS}
        for pid in slot_ids:
            actual_counts[players_meta[pid]["element_type"]] += 1
        composition_ok = actual_counts == EXPECTED_COUNTS
        if not composition_ok:
            expected_str = ", ".join(f"{n} {POSITION_NAMES[p]}" for p, n in EXPECTED_COUNTS.items())
            actual_str = ", ".join(f"{n} {POSITION_NAMES[p]}" for p, n in actual_counts.items())
            st.error(
                f"This doesn't add up to a valid squad: expected {expected_str}, got {actual_str}. "
                "That usually means a name matched the wrong player -- check the table above and the "
                "corrections below for anyone whose position looks off, then fix and re-check."
            )

    ready = all_names_resolved and composition_ok

    if st.button("Save as my squad", disabled=not ready):
        n_starting = len(parsed["starting"])
        squad_ids = slot_ids  # already position-aligned to all_names, no name-collision risk
        starting_ids = slot_ids[:n_starting]

        def _id_for(name: str) -> int | None:
            try:
                return slot_ids[all_names.index(name)]
            except ValueError:
                return None

        captain_id = _id_for(parsed["captain"])
        vc_id = _id_for(parsed["vice_captain"])

        conn = db.get_connection()
        conn.execute(
            """
            INSERT INTO team_state (team_key, gw, squad_json, starting_json, captain_id, vice_captain_id, updated_at)
            VALUES ('user_team', (SELECT COALESCE((SELECT id FROM gameweeks WHERE is_next=1), 1)), ?, ?, ?, ?, ?)
            ON CONFLICT(team_key) DO UPDATE SET
                squad_json=excluded.squad_json, starting_json=excluded.starting_json,
                captain_id=excluded.captain_id, vice_captain_id=excluded.vice_captain_id, updated_at=excluded.updated_at
            """,
            (json.dumps(squad_ids), json.dumps(starting_ids), captain_id, vc_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
        st.session_state.saved_squad_ids = squad_ids
        st.success("Saved. Suggestions below now reflect your real squad.")
    elif not ready and still_unresolved:
        st.caption(f"Fix the unmatched name(s) above before saving: {', '.join(still_unresolved)}")

st.divider()
st.subheader("Transfer suggestions for your saved squad")

conn = db.get_connection()
row = conn.execute("SELECT squad_json FROM team_state WHERE team_key='user_team'").fetchone()
conn.close()

if not row:
    st.info("Upload and save a squad above to see suggestions here.")
else:
    squad_ids = json.loads(row[0])
    with st.spinner("Computing suggestions from all 4 models..."):
        model_result = compute_all_model_scores(qualitative_backend="ollama")
        scores = model_result["scores"]
        if scores:
            scores["ensemble"] = compute_ensemble_scores(scores)

    if model_result["errors"]:
        for key, err in model_result["errors"].items():
            st.caption(f"⚠️ {MODEL_LABELS.get(key, key)} unavailable: {err}")

    explainer(
        "**Rank gain** is a percentile shift on a 0-1 scale (that model's worst pick to its best) -- "
        "comparable across models, so a Model 2 swap and a Model 3 swap can be judged side by side. "
        "**Native gain** is that model's own raw units (e.g. Model 2's is literally predicted points) -- "
        "useful within that model, but not comparable to another model's native gain.",
        label="Why two gain columns?",
    )

    players_meta = load_players_meta()
    for key in ["model1", "model2", "model3", "ensemble"]:
        if key not in scores:
            continue
        st.markdown(f"**{MODEL_LABELS[key]}**")
        swaps = best_single_swaps(squad_ids, scores[key], players_meta, top_n=3)
        if not swaps:
            st.caption("No improving swap found.")
            continue
        st.dataframe(
            [{"Out": s["out_name"], "In": s["in_name"], "Price change": f"£{(s['in_cost']-s['out_cost'])/10:+.1f}m",
              "Rank gain (comparable)": f"{s['gain']:+.3f}", "Native gain (this model's units)": f"{s['raw_gain']:+.3f}"} for s in swaps],
            use_container_width=True, hide_index=True,
        )
