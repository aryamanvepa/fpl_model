"""Parses an FPL squad screenshot into structured data via a single, cheap
Claude API call -- the ONLY place in this project that calls a paid API for
this feature. Everything downstream (matching names to player ids, scoring,
optimizing, suggesting transfers) is plain local computation, no API calls.

Kept deliberately cheap:
  - the image is downscaled before sending (fewer image tokens)
  - uses the smallest/cheapest available model (Haiku), not Sonnet/Opus
  - the prompt asks for nothing but compact JSON, capping output tokens low
  - one call per screenshot, no retries-with-bigger-model, no agentic loop
"""

import base64
import io
import json
import os

import requests

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_VISION_MODEL", "claude-haiku-4-5-20251001")
MAX_IMAGE_DIM = 1024  # px -- plenty to read player names off a squad screenshot

PROMPT = """This is a screenshot of a Fantasy Premier League (FPL) squad. There are ALWAYS \
exactly 15 players: 11 in the starting lineup (pitch view) -- 1 goalkeeper, and 10 outfield \
players in whatever formation is shown -- plus 4 on the bench -- always exactly 1 goalkeeper \
and 3 outfield players, usually shown as a row below/separate from the pitch. The bench \
goalkeeper is easy to miss (often smaller or a different layout) -- double check it's included.

Use the short display name as shown on screen (e.g. "Haaland", "B.Fernandes", "Salah"). Note \
which player has a captain badge (C) and which has vice-captain (VC). If a name is genuinely \
illegible, still include your best-effort reading rather than omitting the player -- every \
one of the 15 slots must appear in your answer.

Respond with ONLY this JSON shape, no other text:
{"starting": ["name", ...11 names...], "bench": ["name", ...4 names...], "captain": "name", "vice_captain": "name"}"""


def _downscale(image_bytes: bytes) -> tuple[bytes, str]:
    if not _PIL_AVAILABLE:
        return image_bytes, "image/png"
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_DIM:
        ratio = MAX_IMAGE_DIM / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)  # jpeg + modest quality keeps tokens down further
    return buf.getvalue(), "image/jpeg"


def parse_team_screenshot(image_bytes: bytes, api_key: str | None = None) -> dict:
    """Returns {"starting": [...], "bench": [...], "captain": str, "vice_captain": str}."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it in your environment to use screenshot parsing.")

    image_data, media_type = _downscale(image_bytes)
    b64 = base64.standard_b64encode(image_data).decode("ascii")

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 400,  # just enough for 15 short names + 2 tags as JSON
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"]

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"Couldn't find JSON in the model's response: {text[:200]}")
    parsed = json.loads(text[start:end + 1])

    for key in ("starting", "bench", "captain", "vice_captain"):
        if key not in parsed:
            raise RuntimeError(f"Model response missing '{key}': {parsed}")
    return parsed
