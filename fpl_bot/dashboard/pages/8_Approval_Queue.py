import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from fpl_bot import db
from fpl_bot.dashboard.utils import MODEL_LABELS, page_header
from fpl_bot.ensemble.team_state import apply_team, list_pending_approvals, resolve_approval

st.set_page_config(page_title="Approval Queue", page_icon="✅", layout="wide")
page_header("Approval Queue", "Mixed-autonomy rule: captain/bench changes auto-apply. Squad composition changes (drafts, transfers) wait here.")

pending = list_pending_approvals()

if not pending:
    st.success("Nothing pending right now.")
else:
    for item in pending:
        label = MODEL_LABELS.get(item["team_key"], item["team_key"])
        with st.container(border=True):
            if item["kind"] == "initial_draft":
                p = item["payload"]
                st.markdown(f"**{label}** -- GW{item['gw']} -- DRAFT initial squad "
                            f"({len(p['squad'])} players, £{p['cost']/10:.1f}m)")
            elif item["kind"] == "transfer":
                p = item["payload"]
                st.markdown(f"**{label}** -- GW{item['gw']} -- TRANSFER "
                            f"{len(p['out'])} out / {len(p['in'])} in")
            else:
                st.markdown(f"**{label}** -- GW{item['gw']} -- {item['kind']}")
            st.caption(f"Queued {item['created_at']}")

            c1, c2, _ = st.columns([1, 1, 4])
            if c1.button("Approve", key=f"approve_{item['id']}"):
                result = apply_team(item["team_key"], item["gw"])
                resolve_approval(item["id"], approved=True)
                st.success(f"Applied -- {label} now holds {len(result.squad)} players, captain "
                           f"{result.captain.web_name if result.captain else '?'}.")
                st.rerun()
            if c2.button("Reject", key=f"reject_{item['id']}"):
                resolve_approval(item["id"], approved=False)
                st.info("Rejected -- will be re-proposed next time the digest runs, if still applicable.")
                st.rerun()

st.divider()
st.subheader("History")
conn = db.get_connection()
try:
    resolved = conn.execute(
        "SELECT team_key, gw, kind, status, created_at, resolved_at FROM pending_approvals "
        "WHERE status != 'pending' ORDER BY id DESC LIMIT 50"
    ).fetchall()
finally:
    conn.close()

if resolved:
    import pandas as pd
    st.dataframe(
        pd.DataFrame(resolved, columns=["Team", "GW", "Kind", "Status", "Created", "Resolved"]),
        use_container_width=True, hide_index=True,
    )
else:
    st.caption("No resolved items yet.")
