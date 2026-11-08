from promorank.parser import parse_episode_text, parse_timestamp


def test_parse_timestamp():
    assert parse_timestamp("00:01.50") == 1.5
    assert parse_timestamp("02:10.25") == 130.25
    assert parse_timestamp("01:02:03.50") == 3723.5


def test_parse_episode_text_extracts_blocks():
    text = '''EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A woman discovers a secret.
CONTINUITY NOTES: Test continuity.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. ROOM – NIGHT
Scene start: 00:00.00
Scene end: 01:10.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:03.00)
Close-up on the door.

**WOMAN (Terrified):**
Start: (00:00.50) | End: (00:02.50)
"Who is there?"

**ACTION:**
Start: (00:03.00) | End: (00:05.00)
The door slams shut.
'''
    ep = parse_episode_text(text, "episode-007.txt")
    assert ep.number == 7
    assert ep.duration == 70.0
    assert len(ep.scenes) == 1
    assert len(ep.events) == 3
    dialogue = [event for event in ep.events if event.event_type == "dialogue"][0]
    assert dialogue.speaker == "WOMAN"
    assert "Who is there?" in dialogue.text
