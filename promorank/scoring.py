"""Transparent, deterministic promo candidate generation and scoring.

Two rules run through this module:

* A line only counts if the viewer hears all of it, so every scorer works on events
  that sit fully inside the clip.
* Cuts land on shot boundaries and never inside a spoken line or an action.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import NamedTuple

from .parser import Episode, Event


# Stakes that are not tied to one setting, so a modern show keeps its hook signal.
MODERN_STAKE_WORDS = {
    "ceo", "boss", "company", "fired", "contract", "shares", "bankrupt", "billionaire",
    "money", "debt", "affair", "cheating", "cheated", "mistress", "fiance", "fiancee",
    "engaged", "baby", "son", "daughter", "police", "lawyer", "hospital", "dna",
    "blackmail", "gun", "evidence", "ruined", "pretending", "stranger",
}
HOOK_WORDS = {
    "betray", "betrayed", "poison", "murder", "kill", "dead", "alive", "secret",
    "truth", "lie", "lying", "marry", "marrying", "wife", "husband", "wedding",
    "pregnant", "father", "mother", "sister", "brother", "duke", "king", "queen",
    "arrest", "prison", "blood", "revenge", "expose", "confess", "confession", "steal",
    "stole", "stolen", "destroy", "ruin", "leave", "choose", "promise", "threat",
    "threaten", "witness", "guilty", "innocent", "divorce", "inheritance", "heir",
    "crown", "title", "banish", "exile", "attack", "weapon", "sword", "fire", "save",
    "rescue", "scandal", "traitor", "decree", "council", "court",
} | MODERN_STAKE_WORDS
CLIFF_WORDS = {
    "but", "because", "unless", "until", "before", "tomorrow", "tonight",
    "secret", "truth", "alive", "dead", "kill", "murder", "poison", "arrest", "choose",
    "prove", "witness", "council", "king", "hearing", "decree", "never", "wait", "stop",
    "if", "then", "now", "return", "back", "reveal", "confess",
    "marrying", "married", "pregnant", "heir", "betrayed", "revenge", "divorce",
    "fired", "bankrupt", "affair", "blackmail",
}
# Multi-word cues cannot match a single-word set, so they are matched as phrases.
CLIFF_PHRASES = ("ten days", "one condition", "one day", "by morning", "too late")
EMOTION_WORDS = {
    "furious", "desperate", "screaming", "shouting", "weeping", "sobbing", "terrified",
    "hysterical", "venomous", "threatening", "devastated", "heartbroken", "betrayed",
    "cold", "cruel", "menacing", "ominous", "deadly", "frantic", "pleading", "defiant",
    "resolute", "shocked", "horrified", "angry", "rage", "panic", "fear", "tears",
}
HOOK_PHRASES = (
    "turn around", "look at me", "how dare you", "you are under arrest", "you are dead",
    "who are you", "what have you done", "you betrayed me", "you killed", "i will kill",
    "i will destroy", "help me", "do not move", "get out", "not anymore", "i cannot marry",
    "i did not", "she is alive", "he is alive", "i know the truth", "stop right there",
    "you have no idea", "this is my", "i can explain", "why are you", "what do you mean",
)
ACTION_WORDS = {
    "slaps", "slap", "strikes", "strike", "punches", "grabs", "shoves", "collapses",
    "falls", "gasp", "gasps", "screams", "screaming", "draws", "raises", "points",
    "chokes", "trembles", "trembling", "stumbles", "kneels", "drops", "reveals",
    "enters", "storms", "lunges", "attacks", "blood", "weapon", "sword", "gun",
}
THREAT_WORDS = {
    "kill", "destroy", "ruin", "arrest", "banish", "exile", "threat", "threaten",
    "revenge", "expose", "fired", "blackmail", "regret", "die", "warn", "warning",
}
REVEAL_WORDS = {
    "truth", "secret", "alive", "dead", "pregnant", "heir", "marrying", "married",
    "confess", "confession", "traitor", "betrayed", "affair", "dna",
}

WEIGHTS = {
    "hook": 30.0,
    "cliffhanger": 30.0,
    "standalone": 20.0,
    "emotion": 15.0,
    "duration": 5.0,
}

# Screenplay timecodes are given to 0.01s; this only absorbs rounding, not real overlap.
CUT_TOLERANCE = 0.05
HOOK_WINDOW_SECONDS = 3.0
MIN_DURATION = 20.0
MAX_DURATION = 60.0
# Below this many camera blocks an episode has no usable shot list to cut on.
MIN_CAMERA_BLOCKS = 3
# How far past the cliffhanger line a beat-driven window may run to reach a clean cut.
MAX_REACTION_SECONDS = 8.0
# Two picks may share at most this fraction of the shorter clip.
MAX_SHARED_FRACTION = 0.3

NO_HOOK_TEXT = "No dialogue or action in the opening 3 seconds."
_OPENS_MID_THOUGHT = re.compile(r"""^["“'\s]*(\.\.\.|…|but\b|and\b)""", re.I)
_TITLE_RE = re.compile(r"^(lady|lord|sir|the|young|old|dame|madam|master)\s+", re.I)
_CLIFF_TIMING_RE = re.compile(r"\b(if|unless|until|before|when|tomorrow|tonight|ten days|one day)\b", re.I)
_CLIFF_STAKES_RE = re.compile(
    r"\b(secret|truth|alive|dead|poison|murder|kill|arrest|traitor|heir|witness|confess|prove"
    r"|marry|marrying|married|pregnant|son|daughter|wife|husband|fired|bankrupt|affair|blackmail|dna)\b",
    re.I,
)
_CLIFF_RESOLVED_RE = re.compile(
    r"\b(it's okay|i forgive you|we are safe|we're safe|thank you|i love you|all is well|we did it|i accept)\b",
    re.I,
)
_STANDALONE_INTRO_RE = re.compile(
    r"\b(my (wife|husband|mother|father|sister|brother)|this is|i am|i'm|you are|your (wife|husband|daughter|son))\b"
)
_STANDALONE_DANGLING_RE = re.compile(r"^(and|but|so|because|then|he|she|it|they|him|her|that)\b")
_DANGER_TERMS = {
    "slap", "slaps", "strike", "strikes", "gun", "sword", "blood", "attack", "attacks",
    "threat", "weapon", "collapse", "collapses", "scream", "screams", "explosion",
}


@dataclass
class Candidate:
    candidate_id: str
    episode: int
    start: float
    end: float
    duration: float
    hook_score: float
    cliffhanger_score: float
    standalone_score: float
    emotion_score: float
    duration_score: float
    heuristic_score: float
    final_score: float
    hook: str
    cliffhanger: str
    rationale: str
    dialogue_excerpt: str
    scene_numbers: list[int]
    llm_evaluation: dict | None = None
    hook_at: float = 0.0
    hook_type: str = "none"
    at_episode_end: bool = False
    candidate_source: str = "heuristic"
    premise_value: float | None = None
    spoiler_risk: float | None = None
    beat_reason: str = ""
    pairwise_net: int = 0
    pairwise_games: int = 0
    premise_adjustment: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Hook(NamedTuple):
    score: float
    text: str
    offset: float
    kind: str
    speaker: str


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def _inside(event: Event, start: float, end: float, tolerance: float = CUT_TOLERANCE) -> bool:
    """True when the whole event plays inside the clip."""
    return event.start >= start - tolerance and event.end <= end + tolerance


def clip_events(events: list[Event], start: float, end: float) -> list[Event]:
    return [e for e in events if _inside(e, start, end)]


def _clamp(value: float, low: float = 0.0, high: float = 5.0) -> float:
    return max(low, min(high, value))


def _event_text(events: list[Event]) -> str:
    return " ".join(e.text for e in events if e.text)


def _clean_quote(text: str, limit: int = 260) -> str:
    return text.replace("\n", " ").strip()[:limit]


def _normalize_speaker(speaker: str) -> str:
    """Treat "LADY JULIANA" and "JULIANA" as the same character."""
    name = speaker.strip().upper()
    stripped = _TITLE_RE.sub("", name).strip()
    return stripped or name


def _speaker_count(events: list[Event]) -> int:
    return len({_normalize_speaker(e.speaker) for e in events if e.event_type == "dialogue" and e.speaker})


def _straddles(event: Event, t: float, tolerance: float) -> bool:
    return event.start < t - tolerance and event.end > t + tolerance


def cuts_through_dialogue(events: list[Event], t: float, tolerance: float = CUT_TOLERANCE) -> bool:
    """True when a cut at ``t`` would land in the middle of a spoken line."""
    return any(e.event_type == "dialogue" and _straddles(e, t, tolerance) for e in events)


def is_clean_cut(events: list[Event], t: float, tolerance: float = CUT_TOLERANCE) -> bool:
    """A cut must not interrupt a spoken line or a physical action such as a slap."""
    return not any(e.event_type in {"dialogue", "action"} and _straddles(e, t, tolerance) for e in events)


def clean_cuts(episode: Episode) -> tuple[list[float], list[float]]:
    """Allowed start and end points: shot boundaries, or beat edges if there is no shot list."""
    cameras = [e for e in episode.events if e.event_type == "camera"]
    source = cameras if len(cameras) >= MIN_CAMERA_BLOCKS else [e for e in episode.events if e.event_type != "camera"]

    def usable(t: float) -> bool:
        # Camera blocks can overlap each other, so one shot's edge may sit inside another shot.
        return is_clean_cut(episode.events, t) and not any(_straddles(c, t, CUT_TOLERANCE) for c in cameras)

    starts = sorted({round(e.start, 2) for e in source if usable(e.start)})
    ends = sorted({round(e.end, 2) for e in source if usable(e.end)})
    return starts, ends


def _classify_hook(beat: Event) -> str:
    if beat.event_type in {"action", "sound_fx"}:
        return "visual"
    words = _words(beat.text)
    if "?" in beat.text:
        return "question"
    if words & THREAT_WORDS:
        return "threat"
    if words & REVEAL_WORDS:
        return "reveal"
    return "statement"


def score_hook(events: list[Event], start: float, end: float = float("inf")) -> Hook:
    """Score what the viewer gets in the first 3 seconds, and report the first beat."""
    opening = sorted(
        (
            e for e in events
            if e.event_type != "camera" and e.text.strip()
            and e.start >= start - CUT_TOLERANCE
            and e.start < start + HOOK_WINDOW_SECONDS
            and e.end <= end + CUT_TOLERANCE
        ),
        key=lambda e: (e.start, e.end),
    )
    if not opening:
        return Hook(0.0, NO_HOOK_TEXT, 0.0, "none", "")

    dialogue = [e for e in opening if e.event_type == "dialogue"]
    text = _event_text(opening)
    words = _words(text)
    score = 1.1
    score += min(1.7, 0.34 * len(words & HOOK_WORDS))
    score += min(0.9, 0.3 * sum(phrase in text.lower() for phrase in HOOK_PHRASES))
    if "?" in text:
        score += 0.75
    if "!" in text:
        score += 0.5
    if len(dialogue) >= 1:
        score += 0.45
    if len(dialogue) >= 2:
        score += 0.25
    labels = " ".join(e.label.lower() for e in opening)
    score += min(0.8, 0.18 * sum(term in labels for term in EMOTION_WORDS))
    action_text = " ".join(e.text.lower() for e in opening if e.event_type in {"action", "sound_fx"})
    score += min(0.7, 0.18 * sum(term in action_text for term in ACTION_WORDS))

    beat = opening[0]
    return Hook(
        _clamp(score),
        _clean_quote(beat.text),
        round(max(0.0, beat.start - start), 2),
        _classify_hook(beat),
        beat.speaker,
    )


def _last_line(events: list[Event], start: float, end: float) -> Event | None:
    dialogue = [e for e in clip_events(events, start, end) if e.event_type == "dialogue"]
    return max(dialogue, key=lambda e: (e.end, e.start), default=None)


def score_cliffhanger(
    events: list[Event],
    end: float,
    start: float = 0.0,
    episode_end: float | None = None,
) -> tuple[float, str]:
    """Judge the last line the viewer fully hears, plus any action that follows it."""
    inside = clip_events(events, start, end)
    tail_actions = [e for e in inside if e.event_type in {"action", "sound_fx"} and e.end > end - 5.0]
    last_dialogue = _last_line(events, start, end)
    at_episode_end = episode_end is not None and end >= episode_end - 0.5

    if last_dialogue:
        quote = last_dialogue.text.strip()
        words = _words(quote)
        label = last_dialogue.label.lower()
        score = 0.7
        score += min(1.7, 0.42 * (len(words & CLIFF_WORDS) + sum(p in quote.lower() for p in CLIFF_PHRASES)))
        if "?" in quote:
            score += 1.25
        if "!" in quote:
            score += 0.2
        if _CLIFF_TIMING_RE.search(quote):
            score += 0.65
        stakes = bool(_CLIFF_STAKES_RE.search(quote))
        if stakes:
            score += 0.75
        # A short flat statement with stakes is a reveal even without a question mark.
        if stakes and "?" not in quote and len(quote.split()) <= 9:
            score += 0.5
        score += min(0.8, 0.2 * sum(term in label for term in EMOTION_WORDS))

        gap = max(0.0, end - last_dialogue.end)
        held_on_reaction = gap <= MAX_REACTION_SECONDS and any(
            e.event_type in {"action", "sound_fx"} and e.end > last_dialogue.end for e in inside
        )
        if gap <= 0.8:
            score += 0.25
        elif gap > 3.0 and not held_on_reaction and not at_episode_end:
            # Dead air after the last line deflates the ending; a held reaction shot does not.
            score -= min(1.5, 0.35 * (gap - 3.0))
        if _CLIFF_RESOLVED_RE.search(quote):
            score -= 0.8
    else:
        quote = ""
        score = 0.4

    if at_episode_end:
        score += 0.6  # Episodes in this format are written to close on their cliffhanger.
    action_words = _words(" ".join(e.text.lower() for e in tail_actions))
    score += min(0.9, 0.3 * len(action_words & _DANGER_TERMS))
    action_labels = " ".join(e.label.lower() for e in tail_actions)
    score += min(0.4, 0.2 * sum(term in action_labels for term in EMOTION_WORDS))
    return _clamp(score), _clean_quote(quote)


def score_standalone(events: list[Event], start: float, end: float) -> float:
    selected = clip_events(events, start, end)
    dialogue = [e for e in selected if e.event_type == "dialogue"]
    speakers = _speaker_count(selected)
    if not dialogue:
        return 0.5
    score = 1.0
    score += min(1.3, 0.32 * len(dialogue))
    score += min(1.0, 0.35 * max(0, speakers - 1))
    text = _event_text(dialogue).lower()
    if _STANDALONE_INTRO_RE.search(text):
        score += 0.7
    if len(text.split()) >= 45:
        score += 0.4
    first = min(dialogue, key=lambda e: e.start).text.lower().strip(' "')
    if _STANDALONE_DANGLING_RE.match(first):
        score -= 0.45
    if len(dialogue) == 1:
        score -= 0.5
    return _clamp(score)


def score_emotion(events: list[Event], start: float, end: float) -> float:
    selected = clip_events(events, start, end)
    text = " ".join(e.text.lower() + " " + e.label.lower() for e in selected)
    words = _words(text)
    score = 0.8 + min(2.0, 0.3 * len(words & EMOTION_WORDS))
    score += min(1.5, 0.22 * sum(term in text for term in ACTION_WORDS))
    score += min(0.7, 0.1 * sum(mark in text for mark in ("!", "?", "...", "—")))
    return _clamp(score)


def score_duration(duration: float) -> float:
    # Prefer a compact, social-friendly cut while accepting the full 20–60s brief.
    if 35 <= duration <= 45:
        return 5.0
    if 30 <= duration <= 50:
        return 4.5
    if 25 <= duration <= 55:
        return 4.0
    return 3.2


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:05.2f}"


def _shorten(text: str, limit: int = 90) -> str:
    text = text.strip().strip('"“”').strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


_HOOK_DESCRIPTIONS = {
    "question": "a question",
    "threat": "a threat",
    "reveal": "a reveal",
    "visual": "a visual beat",
    "statement": "a direct line",
}


def _why_it_works(hook: Hook, last: Event, cliff_score: float, at_end: bool, speakers: int, scenes: int) -> str:
    """One plain sentence from the clip's actual content, usable without an LLM."""
    who = f" from {hook.speaker}" if hook.speaker else ""
    opener = f'{_HOOK_DESCRIPTIONS.get(hook.kind, "a line")}{who} ("{_shorten(hook.text)}")'
    if at_end:
        closer = "the episode's own final beat"
    elif "?" in last.text:
        closer = "an unanswered question"
    elif cliff_score >= 3.0:
        closer = "a line that raises the stakes"
    else:
        closer = "a line with limited suspense, so check the ending"
    scene_text = "one scene" if scenes <= 1 else f"{scenes} scenes"
    speaker_text = "one speaking character" if speakers == 1 else f"{speakers} speaking characters"
    return (
        f'Opens {hook.offset:.1f}s in on {opener} and closes on {closer}: {last.speaker} says '
        f'"{_shorten(last.text)}" It has {speaker_text} in {scene_text}.'
    )


def _score_candidate(episode: Episode, start: float, end: float) -> Candidate | None:
    duration = round(end - start, 2)
    if duration < MIN_DURATION or duration > MAX_DURATION or end > episode.duration + CUT_TOLERANCE:
        return None
    # A promo that opens or closes mid-sentence or mid-slap is unusable, whatever its score.
    if not is_clean_cut(episode.events, start) or not is_clean_cut(episode.events, end):
        return None
    selected = clip_events(episode.events, start, end)
    dialogue = sorted((e for e in selected if e.event_type == "dialogue"), key=lambda e: e.start)
    if len(dialogue) < 2 or len(_event_text(dialogue).split()) < 12:
        return None
    hook = score_hook(episode.events, start, end)
    if hook.kind == "none" or _OPENS_MID_THOUGHT.match(dialogue[0].text):
        return None

    at_end = end >= episode.duration - 0.5
    cliff_score, cliff = score_cliffhanger(episode.events, end, start=start, episode_end=episode.duration)
    standalone = score_standalone(episode.events, start, end)
    emotion = score_emotion(episode.events, start, end)
    duration_score = score_duration(duration)
    weighted = (
        hook.score * WEIGHTS["hook"] / 5
        + cliff_score * WEIGHTS["cliffhanger"] / 5
        + standalone * WEIGHTS["standalone"] / 5
        + emotion * WEIGHTS["emotion"] / 5
        + duration_score * WEIGHTS["duration"] / 5
    )
    excerpt = " ".join(f"{e.speaker}: {e.text.strip()}" for e in dialogue[:8])
    scene_numbers = sorted({e.scene for e in selected if e.scene})
    last = max(dialogue, key=lambda e: (e.end, e.start))
    rationale = _why_it_works(hook, last, cliff_score, at_end, _speaker_count(selected), len(scene_numbers))
    candidate_id = f"E{episode.number:02d}-{int(round(start*10)):05d}-{int(round(end*10)):05d}"
    return Candidate(
        candidate_id=candidate_id,
        episode=episode.number,
        start=round(start, 2),
        end=round(end, 2),
        duration=duration,
        hook_score=round(hook.score, 2),
        cliffhanger_score=round(cliff_score, 2),
        standalone_score=round(standalone, 2),
        emotion_score=round(emotion, 2),
        duration_score=round(duration_score, 2),
        heuristic_score=round(weighted, 2),
        final_score=round(weighted, 2),
        hook=hook.text,
        cliffhanger=cliff,
        rationale=rationale,
        dialogue_excerpt=excerpt[:1600],
        scene_numbers=scene_numbers,
        hook_at=hook.offset,
        hook_type=hook.kind,
        at_episode_end=at_end,
    )


def _cut_windows(episode: Episode) -> set[tuple[float, float]]:
    starts, ends = clean_cuts(episode)
    return {
        (start, end) for start in starts for end in ends
        if MIN_DURATION <= round(end - start, 2) <= MAX_DURATION
    }


def _grid_windows(
    episode: Episode, durations: tuple[int, ...], stride_seconds: float
) -> set[tuple[float, float]]:
    """Fixed-stride fallback for scripts whose blocks give no usable cut points."""
    max_start = max(0.0, episode.duration - MIN_DURATION)
    windows: set[tuple[float, float]] = set()
    for start in _frange(0.0, max_start, stride_seconds):
        for duration in durations:
            end = round(start + duration, 2)
            if end <= episode.duration + 0.001:
                windows.add((round(start, 2), end))
    return windows


def generate_candidates(
    episodes: list[Episode],
    durations: tuple[int, ...] = (20, 25, 30, 35, 40, 45, 50, 55, 60),
    stride_seconds: float = 2.5,
) -> list[Candidate]:
    """Build every valid 20–60s window between clean cuts in each episode."""
    candidates: list[Candidate] = []
    for episode in episodes:
        if episode.duration < MIN_DURATION:
            continue
        scored = [_score_candidate(episode, s, e) for s, e in sorted(_cut_windows(episode))]
        valid = [c for c in scored if c]
        if not valid:
            scored = [_score_candidate(episode, s, e) for s, e in sorted(_grid_windows(episode, durations, stride_seconds))]
            valid = [c for c in scored if c]
        candidates.extend(valid)
    return sorted(candidates, key=lambda c: c.heuristic_score, reverse=True)


def window_for_lines(episode: Episode, hook_line: Event, cliff_line: Event) -> Candidate | None:
    """Turn two chosen lines into a validated window. Only Python ever sets timestamps.

    Starts at the nearest clean cut that keeps the hook inside the first 3 seconds, and
    ends at the nearest clean cut after the cliffhanger line.
    """
    starts, ends = clean_cuts(episode)
    start_options = sorted(
        (s for s in starts if hook_line.start - HOOK_WINDOW_SECONDS < s <= hook_line.start + CUT_TOLERANCE),
        reverse=True,
    )
    end_options = sorted(
        e for e in ends if cliff_line.end - CUT_TOLERANCE <= e <= cliff_line.end + MAX_REACTION_SECONDS
    )
    for start in start_options:
        for end in end_options:
            candidate = _score_candidate(episode, start, end)
            if candidate:
                return candidate
    return None


def numbered_lines(episode: Episode) -> list[tuple[str, Event]]:
    """Stable line IDs for everything a viewer hears or sees described, excluding camera notes."""
    usable = [e for e in episode.events if e.event_type != "camera" and e.text.strip()]
    return [(f"L{i}", event) for i, event in enumerate(usable, 1)]


def _frange(start: float, stop: float, step: float):
    value = start
    while value <= stop + 1e-9:
        yield value
        value += step


def shared_fraction(a: Candidate, b: Candidate) -> float:
    """How much of the shorter clip the other clip also covers (1.0 means nested)."""
    if a.episode != b.episode:
        return 0.0
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shorter = min(a.end - a.start, b.end - b.start)
    return intersection / shorter if shorter > 0 else 0.0


def diversify(
    candidates: list[Candidate],
    max_results: int = 100,
    max_per_episode: int = 3,
    max_shared: float = MAX_SHARED_FRACTION,
    preselected: list[Candidate] | None = None,
) -> list[Candidate]:
    """Select broad episode coverage first, then fill remaining slots by global score.

    Overlap is measured against the shorter clip, so a 20s clip nested in a 50s clip is
    rejected. ``preselected`` candidates count toward the per-episode and overlap limits
    but are not returned; ``max_results`` is the number of *additional* candidates to choose.
    """
    ordered = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    reserved = list(preselected or [])
    chosen: list[Candidate] = []
    per_episode: dict[int, int] = {}
    for candidate in reserved:
        per_episode[candidate.episode] = per_episode.get(candidate.episode, 0) + 1

    def add_if_valid(candidate: Candidate) -> bool:
        if per_episode.get(candidate.episode, 0) >= max_per_episode:
            return False
        if any(shared_fraction(candidate, other) > max_shared for other in chosen + reserved):
            return False
        chosen.append(candidate)
        per_episode[candidate.episode] = per_episode.get(candidate.episode, 0) + 1
        return True

    # Pass 1: one strongest candidate per episode, so a global top-N cannot hide episodes.
    best_by_episode: dict[int, Candidate] = {}
    for candidate in ordered:
        if candidate.episode not in best_by_episode:
            best_by_episode[candidate.episode] = candidate
    for candidate in sorted(best_by_episode.values(), key=lambda c: c.final_score, reverse=True):
        if len(chosen) >= max_results:
            break
        add_if_valid(candidate)

    # Pass 2: fill the rest with the next strongest non-duplicate candidates.
    for candidate in ordered:
        if len(chosen) >= max_results:
            break
        add_if_valid(candidate)
    return sorted(chosen, key=lambda c: c.final_score, reverse=True)


def timestamp_range(candidate: Candidate) -> str:
    return f"{_format_time(candidate.start)}–{_format_time(candidate.end)}"


def format_range(start: float, end: float) -> str:
    return f"{_format_time(start)}–{_format_time(end)}"


def clip_transcript(episode: Episode, start: float, end: float, max_chars: int = 3000) -> str:
    """What the viewer actually gets: only events fully inside the clip, with clip-relative times.

    Includes action and sound cues (not just dialogue) so a reviewer can judge
    visual hooks such as a slap or a collapse. Camera notes are left out to save space.
    """
    lines = []
    for e in clip_events(episode.events, start, end):
        if e.event_type == "camera" or not e.text.strip():
            continue
        offset = max(0.0, e.start - start)
        who = e.label if e.event_type == "dialogue" else e.event_type.upper().replace("_", " ")
        text = " ".join(e.text.split())
        lines.append(f"[{offset:04.1f}s] {who}: {text}")
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        transcript = transcript[: max_chars - 15].rsplit("\n", 1)[0] + "\n[...truncated]"
    return transcript


def line_after_cut(episode: Episode, end: float) -> str:
    """The first line the viewer does NOT get, so a judge can tell if the question stays open."""
    later = sorted(
        (e for e in episode.events if e.event_type != "camera" and e.text.strip() and e.start >= end - CUT_TOLERANCE),
        key=lambda e: (e.start, e.end),
    )
    if not later:
        return "(the episode ends here)"
    who = later[0].label if later[0].event_type == "dialogue" else later[0].event_type.upper().replace("_", " ")
    return f"{who}: {' '.join(later[0].text.split())}"
