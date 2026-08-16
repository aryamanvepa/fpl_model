"""Free, ToS-respecting source fetchers for Model 4's qualitative signal.

Reddit's r/FantasyPL was evaluated and dropped from v1: its public `.json`
endpoint now returns 403 without a registered OAuth app (a credential only
the account owner can create) -- see PLAN.md for the follow-up.
"""

import xml.etree.ElementTree as ET

import requests

from fpl_bot import db

BBC_FOOTBALL_RSS = "http://feeds.bbci.co.uk/sport/football/rss.xml"


def fetch_bbc_football_headlines(limit: int = 15) -> list[dict]:
    """Recent football headlines + summaries from the BBC Sport RSS feed."""
    resp = requests.get(BBC_FOOTBALL_RSS, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    items = []
    for item in root.findall("./channel/item")[:limit]:
        title = item.findtext("title") or ""
        description = item.findtext("description") or ""
        link = item.findtext("link") or ""
        items.append({"title": title.strip(), "summary": description.strip(), "url": link.strip()})
    return items


def player_status_notes(player_ids: list[int]) -> dict[int, str]:
    """Official FPL-provided status text (injuries/suspensions/doubts) for
    the given players -- already ingested from the bootstrap API, no fetch needed."""
    if not player_ids:
        return {}
    conn = db.get_connection()
    try:
        placeholders = ",".join("?" * len(player_ids))
        rows = conn.execute(
            f"SELECT id, news FROM players WHERE id IN ({placeholders}) AND news != ''",
            player_ids,
        ).fetchall()
    finally:
        conn.close()
    return {row[0]: row[1] for row in rows}
