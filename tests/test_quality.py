"""Regression tests for the output audit, including checks on the real pack when present."""
import os
from pathlib import Path

import pytest

from promorank.curation import select_post_first
from promorank.parser import Episode, Event, load_zip_path, parse_episode_text
from promorank.pipeline import _apply_premise, run_pipeline
from promorank.quality import _check_gold, load_gold, report_to_markdown
from promorank.scoring import (
    Candidate,
    clean_cuts,
    clip_transcript,
    diversify,
    generate_candidates,
    line_after_cut,
    score_cliffhanger,
    score_hook,
    shared_fraction,
)

PACK = Path(os.environ.get("PROMORANK_PACK", Path(__file__).resolve().parents[1] / "scripts.zip"))


def _candidate(episode, start, end, score):
    return Candidate(
        candidate_id=f"E{episode}-{start}-{end}", episode=episode, start=start, end=end, duration=end - start,
        hook_score=3, cliffhanger_score=3, standalone_score=3, emotion_score=3, duration_score=3,
        heuristic_score=score, final_score=score, hook="", cliffhanger="", rationale="", dialogue_excerpt="",
        scene_numbers=[],
    )


def _line(start, end, text="A line of dialogue here.", speaker="MAN"):
    return Event(1, 1, "", speaker, speaker, "dialogue", start, end, text)


def _camera(start, end):
    return f"**CAMERA:**\nStart: (00:{start:02d}.00) | End: (00:{end:02d}.00)\nA shot.\n\n"


def test_line_partly_outside_the_clip_is_not_the_hook_or_cliffhanger():
    events = [_line(8.0, 10.2, "Cut off opening line."), _line(11.0, 13.0, "Fully heard line."),
              _line(30.0, 33.0, "Last whole line."), _line(39.8, 42.0, "Cut off closing line.")]
    assert score_hook(events, 10.0, 40.0).text == "Fully heard line."
    _, closing = score_cliffhanger(events, 40.0, start=10.0)
    assert closing == "Last whole line."


def test_hook_reports_the_first_beat_and_when_it_happens():
    events = [Event(1, 1, "", "ACTION", "", "action", 10.4, 12.0, "She slaps him."),
              _line(11.0, 13.0, "A much longer line of dialogue that used to win the hook field.")]
    hook = score_hook(events, 10.0, 40.0)
    assert (hook.text, hook.offset, hook.kind) == ("She slaps him.", 0.4, "visual")


def test_transcript_only_contains_lines_the_viewer_gets():
    text = "EPISODE NUMBER: Episode 1\nSCENE 01\nScene heading: X\nScene start: 00:00.00\nScene end: 00:50.00\n\n"
    text += '**MAN:**\nStart: (00:02.00) | End: (00:06.00)\n"Clipped line."\n\n'
    text += '**WOMAN:**\nStart: (00:06.00) | End: (00:09.00)\n"Kept line."\n\n'
    text += '**MAN:**\nStart: (00:20.00) | End: (00:24.00)\n"Next line."\n'
    episode = parse_episode_text(text, "episode-001.txt")
    transcript = clip_transcript(episode, 5.0, 12.0)
    assert "Kept line" in transcript and "Clipped line" not in transcript
    assert "Next line" in line_after_cut(episode, 12.0)
    assert line_after_cut(episode, 50.0) == "(the episode ends here)"


def test_nested_clip_is_rejected_even_though_its_iou_is_low():
    long_clip, nested = _candidate(14, 5.0, 55.0, 80), _candidate(14, 5.0, 25.0, 79)
    assert shared_fraction(long_clip, nested) == 1.0
    assert diversify([long_clip, nested], max_results=5) == [long_clip]


def test_small_overlap_is_allowed():
    first, second = _candidate(1, 0.0, 30.0, 80), _candidate(1, 25.0, 55.0, 79)
    assert len(diversify([first, second], max_results=5)) == 2


def test_cuts_land_on_shot_edges_and_never_inside_an_action():
    text = "EPISODE NUMBER: Episode 1\nSCENE 01\nScene heading: X\nScene start: 00:00.00\nScene end: 00:40.00\n\n"
    for start, end in ((0, 10), (10, 20), (20, 30), (30, 40)):
        text += _camera(start, end)
    text += '**MAN:**\nStart: (00:01.00) | End: (00:05.00)\n"You betrayed me and everyone in this family knows it."\n\n'
    text += '**ACTION:**\nStart: (00:18.00) | End: (00:22.00)\nShe slaps him.\n\n'
    text += '**WOMAN:**\nStart: (00:31.00) | End: (00:38.00)\n"Then tell me the truth before it is too late."\n'
    episode = parse_episode_text(text, "episode-001.txt")
    edges = {0.0, 10.0, 20.0, 30.0, 40.0}
    candidates = generate_candidates([episode])
    assert candidates
    for candidate in candidates:
        assert candidate.start in edges and candidate.end in edges
        assert candidate.start != 20.0 and candidate.end != 20.0  # 20s is inside the slap.


def test_flat_reveal_at_the_end_is_not_scored_as_a_weak_ending():
    reveal = [_line(20.0, 22.0, "Because she is my wife.")]
    plain = [_line(20.0, 22.0, "I walked over to the market.")]
    assert score_cliffhanger(reveal, 22.0)[0] > score_cliffhanger(plain, 22.0)[0] + 1.0


def test_closing_reaction_shot_is_not_penalized_as_silence():
    line = _line(20.0, 22.0, "Lady Anna... take my hand.")
    reaction = Event(1, 1, "", "ACTION", "", "action", 22.0, 28.0, "She looks at him in tears.")
    held = score_cliffhanger([line, reaction], 28.0, start=0.0, episode_end=28.0)[0]
    dead_air = score_cliffhanger([line], 28.0, start=0.0)[0]
    assert held > dead_air


def test_parser_warns_when_a_block_format_drifts():
    text = "EPISODE NUMBER: Episode 1\n\n**MAN:**\nBegin 00:01 to 00:03\n\"Different format.\"\n"
    episode = parse_episode_text(text, "episode-001.txt")
    assert any("headers" in warning for warning in episode.parse_warnings)
    assert any("no timestamped blocks" in warning for warning in episode.parse_warnings)


def test_post_first_favours_early_episodes_when_scores_tie():
    episodes = [Episode(n, f"{n}.txt", "A woman is betrayed at her wedding.", "", 100.0, [], []) for n in (1, 50)]

    def promo(rank, episode, kind):
        return {"rank": rank, "candidate_id": str(rank), "episode": episode, "time_range": "", "segments": [],
                "final_score": 70.0, "hook_type": kind, "dialogue_excerpt": "", "hook": "", "cliffhanger": ""}
    promos = [promo(1, 50, "question"), promo(2, 1, "threat")]
    tier = select_post_first(promos, episodes)
    assert [t["episode"] for t in tier] == [1, 50]
    assert promos[1]["post_first_rank"] == 1
    assert all(t["reason"].endswith(".") for t in tier)


def _launch_promo(rank, episode, kind, premise=None, spoiler=None, score=70.0):
    return {"rank": rank, "candidate_id": str(rank), "episode": episode, "time_range": "", "segments": [],
            "final_score": score, "hook_type": kind, "dialogue_excerpt": "", "hook": "", "cliffhanger": "",
            "premise_value": premise, "spoiler_risk": spoiler}


def test_post_first_never_mixes_llm_ratings_with_logline_estimates():
    episodes = [Episode(n, f"{n}.txt", "A woman is betrayed at her wedding.", "", 100.0, [], []) for n in (1, 2, 3)]
    # Promo 3 has no LLM rating, and its text would earn a high logline estimate.
    promos = [_launch_promo(1, 1, "question", 4.0), _launch_promo(2, 2, "threat", 3.0),
              _launch_promo(3, 3, "visual", score=90.0)]
    promos[2]["dialogue_excerpt"] = "woman betrayed wedding"
    tier = select_post_first(promos, episodes, count=2)
    assert [t["episode"] for t in tier] == [1, 2]
    assert all("LLM judgment" in t["reason"] for t in tier)


def test_post_first_spoiler_risk_follows_story_position_not_the_llm_rating():
    episodes = [Episode(n, f"{n}.txt", "", "", 100.0, [], []) for n in (1, 25, 50)]
    promos = [_launch_promo(1, 1, "question", 5.0, spoiler=4.0), _launch_promo(2, 50, "threat", 5.0, spoiler=1.0)]
    tier = select_post_first(promos, episodes, count=2)
    assert [(t["episode"], t["spoiler_risk"]) for t in tier] == [(1, 1.0), (50, 5.0)]
    assert "low spoiler risk" in tier[0]["reason"] and "high spoiler risk" in tier[1]["reason"]


def test_premise_rating_moves_the_score_relative_to_the_lists_average():
    strong, weak, unrated = _candidate(1, 0.0, 30.0, 70.0), _candidate(2, 0.0, 30.0, 70.0), _candidate(3, 0.0, 30.0, 70.0)
    strong.premise_value, weak.premise_value = 5.0, 3.0
    ranked = _apply_premise([weak, unrated, strong])
    assert ranked[0] is strong and ranked[-1] is weak
    assert (strong.final_score, weak.final_score, unrated.final_score) == (73.0, 67.0, 70.0)


def test_gold_list_measures_recall_and_finds_the_missed_window(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text("episode,start,end,note\n1,00:10.00,40,picked\n2,0,30,missed\n", encoding="utf-8")
    gold = load_gold(path)
    promo = {"rank": 7, "episode": 1, "start": 12.0, "end": 42.0}
    result = _check_gold(gold, [promo], [_candidate(2, 1.0, 29.0, 50.0), _candidate(1, 12.0, 42.0, 60.0)])
    assert (result["picked"], result["total"], result["recall"]) == (1, 2, 0.5)
    assert result["items"][0]["picked_rank"] == 7
    assert result["items"][1]["best_candidate_rank"] == 2


def test_overlapping_camera_blocks_do_not_create_mid_shot_cuts():
    text = "EPISODE NUMBER: Episode 1\nSCENE 01\nScene heading: X\nScene start: 00:00.00\nScene end: 00:40.00\n\n"
    for start, end in ((0, 10), (8, 20), (20, 30)):
        text += _camera(start, end)
    starts, ends = clean_cuts(parse_episode_text(text, "episode-001.txt"))
    assert 8.0 not in starts and 10.0 not in ends


@pytest.fixture(scope="module")
def real_results():
    if not PACK.exists():
        pytest.skip("Confidential screenplay pack is not available.")
    episodes = load_zip_path(PACK)
    return run_pipeline(episodes), episodes


def test_real_pack_has_no_hook_or_cliffhanger_outside_its_clip(real_results):
    integrity = real_results[0]["quality_report"]["line_integrity"]
    assert integrity["hook_line_outside_clip"] == 0
    assert integrity["cliffhanger_line_outside_clip"] == 0
    assert integrity["hook_line_unverified"] == 0
    assert integrity["cliffhanger_line_unverified"] == 0


def test_real_pack_has_no_nested_copies(real_results):
    overlap = real_results[0]["quality_report"]["same_episode_overlap"]
    assert overlap["nested_pairs"] == 0
    assert overlap["pairs_over_30_percent_shared"] == 0


def test_real_pack_cuts_are_clean(real_results):
    cuts = real_results[0]["quality_report"]["cuts"]
    assert all(value == 0 for value in cuts.values()), cuts


def test_real_pack_covers_every_episode_and_reports(real_results):
    results, episodes = real_results
    assert {p["episode"] for p in results["ranked_promos"]} == {e.number for e in episodes}
    assert all(p["segments"] == [{"episode": p["episode"], "start": p["start"], "end": p["end"]}] for p in results["ranked_promos"])
    assert results["dataset"]["parse_warnings"] == []
    assert 5 <= len(results["post_first"]) <= 10
    assert "quality report" in report_to_markdown(results).lower()
