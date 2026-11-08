from promorank.parser import parse_episode_text
from promorank.scoring import generate_candidates, diversify


def test_candidate_generation_returns_valid_windows():
    text = '''EPISODE NUMBER: Episode 1
EPISODE LOGLINE: A woman is betrayed.
CONTINUITY NOTES: A test.
--- SCRIPT START ---
SCENE 01
Scene heading: INT. ROOM – NIGHT
Scene start: 00:00.00
Scene end: 01:00.00
**WOMAN (Desperate):**
Start: (00:00.00) | End: (00:03.00)
"You betrayed me! Why would you do this?"

**MAN (Cold):**
Start: (00:10.00) | End: (00:14.00)
"Because the truth is worse than you think."

**WOMAN (Shocked):**
Start: (00:30.00) | End: (00:33.00)
"What truth?"

**MAN (Ominous):**
Start: (00:53.00) | End: (00:59.00)
"Your father is alive."
'''
    episode = parse_episode_text(text, "episode-001.txt")
    candidates = generate_candidates([episode], stride_seconds=5.0)
    assert candidates
    assert all(20 <= c.duration <= 60 for c in candidates)
    assert all(c.end <= episode.duration for c in candidates)
    ranked = diversify(candidates, max_results=5, max_per_episode=2)
    assert len(ranked) <= 2


def test_no_candidate_cuts_through_a_spoken_line():
    from promorank.scoring import cuts_through_dialogue

    text = '''EPISODE NUMBER: Episode 2
--- SCRIPT START ---
SCENE 01
Scene heading: INT. ROOM – NIGHT
Scene start: 00:00.00
Scene end: 01:30.00
''' + "".join(
        f'''
**{"WOMAN" if i % 2 else "MAN"} (Angry):**
Start: (00:{i * 8:02d}.00) | End: (00:{i * 8 + 6:02d}.00)
"Line number {i}, and you know the truth about the secret?"
''' for i in range(11)
    )
    episode = parse_episode_text(text, "episode-002.txt")
    candidates = generate_candidates([episode])
    assert candidates
    for c in candidates:
        assert not cuts_through_dialogue(episode.events, c.start), c.candidate_id
        assert not cuts_through_dialogue(episode.events, c.end), c.candidate_id


def test_multi_word_cliffhanger_phrase_counts():
    from promorank.parser import Event
    from promorank.scoring import score_cliffhanger

    def ending(line):
        return [Event(1, 1, "", "MAN", "MAN", "dialogue", 20.0, 24.0, line)]

    with_phrase, _ = score_cliffhanger(ending("You have ten days to leave."), 24.0)
    without, _ = score_cliffhanger(ending("You have some time to leave."), 24.0)
    assert with_phrase > without


def test_speaker_titles_are_normalized():
    from promorank.scoring import _normalize_speaker

    assert _normalize_speaker("LADY JULIANA") == _normalize_speaker("JULIANA")
    assert _normalize_speaker("LORD") == "LORD"
