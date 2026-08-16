"""Review and act on the pending-approval queue that daily_digest.py builds
up: initial squad drafts and transfers -- the things the mixed-autonomy rule
says need your go-ahead, unlike captain/bench changes which auto-apply.

Usage:
    python -m fpl_bot.scripts.approve_pending                  # interactive, one at a time
    python -m fpl_bot.scripts.approve_pending --list            # just list, no prompts
    python -m fpl_bot.scripts.approve_pending --approve 3       # approve a specific item
    python -m fpl_bot.scripts.approve_pending --reject 3        # reject a specific item
    python -m fpl_bot.scripts.approve_pending --approve-all     # approve everything pending
"""

import argparse
import io
import sys
import warnings

warnings.filterwarnings("ignore")

from fpl_bot.ensemble.models_registry import MODEL_LABELS
from fpl_bot.ensemble.team_state import apply_team, list_pending_approvals, resolve_approval

TEAM_LABELS = {**MODEL_LABELS, "ensemble": "Ensemble (Team E)"}


def _describe(item: dict) -> str:
    label = TEAM_LABELS.get(item["team_key"], item["team_key"])
    if item["kind"] == "initial_draft":
        p = item["payload"]
        return f"[{item['id']}] {label} GW{item['gw']}: DRAFT initial squad ({len(p['squad'])} players, £{p['cost']/10:.1f}m)"
    if item["kind"] == "transfer":
        p = item["payload"]
        return f"[{item['id']}] {label} GW{item['gw']}: TRANSFER {len(p['out'])} out / {len(p['in'])} in"
    return f"[{item['id']}] {label} GW{item['gw']}: {item['kind']}"


def approve(item_id: int) -> None:
    items = {i["id"]: i for i in list_pending_approvals()}
    item = items.get(item_id)
    if item is None:
        print(f"No pending item with id {item_id}.")
        return
    result = apply_team(item["team_key"], item["gw"])
    resolve_approval(item_id, approved=True)
    cap = result.captain.web_name if result.captain else "?"
    print(f"  Applied: {TEAM_LABELS.get(item['team_key'], item['team_key'])} now holds "
          f"{len(result.squad)} players, £{result.total_cost/10:.1f}m, captain {cap}.")


def reject(item_id: int) -> None:
    resolve_approval(item_id, approved=False)
    print(f"  Rejected item {item_id} -- team state left unchanged, will be re-proposed next digest run.")


def run(args) -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if args.approve is not None:
        approve(args.approve)
        return
    if args.reject is not None:
        reject(args.reject)
        return

    pending = list_pending_approvals()
    if not pending:
        print("No pending approvals.")
        return

    if args.approve_all:
        for item in pending:
            approve(item["id"])
        return

    for item in pending:
        print(_describe(item))

    if args.list:
        return

    for item in pending:
        answer = input(f"{_describe(item)}\n  Approve? [y/N/skip]: ").strip().lower()
        if answer == "y":
            approve(item["id"])
        elif answer == "skip":
            continue
        else:
            reject(item["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list pending items without prompting")
    parser.add_argument("--approve", type=int, default=None, help="approve a specific item id")
    parser.add_argument("--reject", type=int, default=None, help="reject a specific item id")
    parser.add_argument("--approve-all", action="store_true", help="approve everything currently pending")
    run(parser.parse_args())
