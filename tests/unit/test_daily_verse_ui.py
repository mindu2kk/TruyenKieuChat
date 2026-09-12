import json
from pathlib import Path

from app.poem_tools import get_range
from app.router import parse_poem_request, route_query


ROOT = Path(__file__).resolve().parents[2]
CHAT_JS = ROOT / "chat_UI" / "static" / "chat_UI" / "kieu-chat.js"


def _daily_verses() -> list[dict]:
    source = CHAT_JS.read_text(encoding="utf-8")
    payload = source.split("const DAILY_VERSES = ", 1)[1].split(";", 1)[0]
    return json.loads(payload)


def test_daily_verses_match_canonical_poem_and_route_by_exact_range():
    verses = _daily_verses()

    assert len(verses) >= 5
    for item in verses:
        start, end = item["lineStart"], item["lineEnd"]
        assert item["lines"].splitlines() == get_range(start, end)
        assert parse_poem_request(item["prompt"]) == ("range", start, end)
        decision = route_query(item["prompt"])
        assert decision.intent == "poem"
        assert decision.flow == "grounded-poem-analysis"
        assert decision.requires_exact_quotes is True


def test_daily_verse_template_displays_canonical_line_range():
    template = (ROOT / "chat_UI" / "templates" / "chat.html").read_text(encoding="utf-8")

    assert "dailyVerse.lineStart" in template
    assert "dailyVerse.lineEnd" in template
