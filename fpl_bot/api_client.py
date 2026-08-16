"""Thin client for the official (public, read-only) FPL API."""

import requests

BASE_URL = "https://fantasy.premierleague.com/api"


def get_bootstrap_static() -> dict:
    """Players, teams, positions, gameweeks — the main reference dataset."""
    resp = requests.get(f"{BASE_URL}/bootstrap-static/", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_fixtures() -> list[dict]:
    resp = requests.get(f"{BASE_URL}/fixtures/", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_element_summary(player_id: int) -> dict:
    """Per-player gameweek history and upcoming fixtures."""
    resp = requests.get(f"{BASE_URL}/element-summary/{player_id}/", timeout=15)
    resp.raise_for_status()
    return resp.json()
