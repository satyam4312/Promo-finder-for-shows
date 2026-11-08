"""Editorial layers on top of the ranking: what to post first, and multi-episode montages."""
from __future__ import annotations

import math
import re
from typing import Any

from .parser import Episode
from .scoring import CUT_TOLERANCE, clean_cuts, clip_events, format_range

_STOPWORDS = {
    "about", "after", "against", "also", "because", "been", "before", "being", "between",
    "both", "during", "first", "from", "have", "into", "only", "over", "that", "their",
    "them", "then", "there", "they", "this", "through", "under", "when", "where", "which",
    "while", "with", "without", "would", "episode", "series", "finds", "arrives",
}
MIN_SEGMENT_SECONDS = 6.0
MAX_SEGMENT_SECONDS = 20.0
MIN_MONTAGE_SECONDS = 20.0
MAX_MONTAGE_SECONDS = 60.0


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) >= 4 and w not in _STOPWORDS}


def _premise_estimate(promo: dict[str, Any], premise_words: set[str]) -> float:
    """Without an LLM, approximate premise value by shared words with the opening loglines."""
    clip_words = _content_words(" ".join((promo["dialogue_excerpt"], promo["hook"], promo["cliffhanger"])))
    return min(5.0, 1.0 + 0.8 * len(clip_words & premise_words))


def _spoiler_label(score: float) -> str:
    return "low" if score <= 2 else "moderate" if score <= 3.5 else "high"


def select_post_first(
    promos: list[dict[str, Any]],
    episodes: list[Episode],
    count: int = 8,
) -> list[dict[str, Any]]:
    """Pick a short launch list and set ``post_first_rank`` on the promos chosen.

    Weighs the promo's score against how well it sells the premise and how much it spoils,
    on the assumption that new viewers land on Ep 1. Spoiler risk comes from story position,
    because the LLM's per-episode spoiler ratings do not change across the series.
    Premise ratings are only compared like with like: LLM ratings when enough promos have
    one, otherwise the logline estimate for every promo.
    Hook types are capped so the set is varied enough to A/B test.
    """
    if not promos or not episodes:
        return []
    order = {e.number: i for i, e in enumerate(sorted(episodes, key=lambda e: e.number))}
    last_index = max(1, len(order) - 1)
    premise_words = _content_words(" ".join(e.logline for e in sorted(episodes, key=lambda e: e.number)[:3]))
    judged = [p for p in promos if p.get("llm_score") is not None]
    rated = [p for p in promos if p.get("premise_value") is not None]
    has_llm_view = len(rated) >= count
    pool = rated if has_llm_view else (judged or promos)

    scored = []
    for promo in pool:
        premise = promo["premise_value"] if has_llm_view else _premise_estimate(promo, premise_words)
        spoiler = 1.0 + 4.0 * order[promo["episode"]] / last_index
        # The premise is weighed here, so its effect on the final score must not count twice.
        base_score = promo["final_score"] - promo.get("premise_adjustment", 0.0)
        launch_score = base_score + 5.0 * (premise - 3.0) - 4.0 * (spoiler - 3.0)
        scored.append((launch_score, promo, premise, spoiler))
    scored.sort(key=lambda row: row[0], reverse=True)

    type_cap = max(2, math.ceil(count / 3))
    chosen: list[tuple] = []
    used_episodes: set[int] = set()
    type_counts: dict[str, int] = {}
    for row in scored:
        promo = row[1]
        if len(chosen) >= count:
            break
        if promo["episode"] in used_episodes or type_counts.get(promo["hook_type"], 0) >= type_cap:
            continue
        chosen.append(row)
        used_episodes.add(promo["episode"])
        type_counts[promo["hook_type"]] = type_counts.get(promo["hook_type"], 0) + 1

    tier = []
    previous_type = None
    for position, (launch_score, promo, premise, spoiler) in enumerate(chosen, 1):
        promo["post_first_rank"] = position
        source = "LLM judgment" if has_llm_view else "estimated from the loglines"
        parts = [
            f"Sells the premise {premise:.0f}/5 ({source})",
            f"{_spoiler_label(spoiler)} spoiler risk at Ep {promo['episode']} of {len(order)}",
            f"opens on a {promo['hook_type']} hook"
            + (f", a different test from the {previous_type} hook before it" if previous_type and previous_type != promo["hook_type"] else ""),
        ]
        previous_type = promo["hook_type"]
        tier.append({
            "post_first_rank": position,
            "rank": promo["rank"],
            "candidate_id": promo["candidate_id"],
            "episode": promo["episode"],
            "time_range": promo["time_range"],
            "segments": promo["segments"],
            "final_score": promo["final_score"],
            "launch_score": round(launch_score, 2),
            "hook_type": promo["hook_type"],
            "premise_value": round(premise, 1),
            "spoiler_risk": round(spoiler, 1),
            "reason": f"{promo.get('why_it_works', '')} {'; '.join(parts)}.".strip(),
        })
    return tier


def _short_segment(episode: Episode, start: float, end: float) -> dict[str, Any] | None:
    """Trim a promo down to its closing beat, starting on a clean cut."""
    dialogue = [e for e in clip_events(episode.events, start, end) if e.event_type == "dialogue"]
    if not dialogue:
        return None
    last = max(dialogue, key=lambda e: (e.end, e.start))
    starts, _ = clean_cuts(episode)
    for cut in sorted(starts, reverse=True):
        if cut > last.start + CUT_TOLERANCE or cut < start - CUT_TOLERANCE:
            continue
        length = end - cut
        if length > MAX_SEGMENT_SECONDS:
            return None
        if length >= MIN_SEGMENT_SECONDS:
            return {
                "episode": episode.number,
                "start": round(cut, 2),
                "end": round(end, 2),
                "time_range": format_range(cut, end),
                "beat": last.text.replace("\n", " ").strip()[:160],
            }
    return None


def build_montages(
    promos: list[dict[str, Any]],
    episodes: list[Episode],
    count: int = 3,
) -> list[dict[str, Any]]:
    """Cross-episode promos: one beat each from the start, middle and end of the series."""
    if len(episodes) < 3 or not promos:
        return []
    ordered = sorted(episodes, key=lambda e: e.number)
    by_number = {e.number: e for e in ordered}
    third_of = {e.number: i * 3 // len(ordered) for i, e in enumerate(ordered)}
    pools: list[list[dict[str, Any]]] = [[], [], []]
    for promo in sorted(promos, key=lambda p: p["final_score"], reverse=True):
        pools[third_of[promo["episode"]]].append(promo)

    segment_sets: list[list[tuple[dict, dict]]] = [[], [], []]
    for third, pool in enumerate(pools):
        for promo in pool:
            segment = _short_segment(by_number[promo["episode"]], promo["start"], promo["end"])
            if segment:
                segment_sets[third].append((promo, segment))

    montages = []
    for index in range(min(count, *(len(s) for s in segment_sets))):
        picks = [segment_sets[third][index] for third in range(3)]
        segments = [segment for _, segment in picks]
        total = round(sum(s["end"] - s["start"] for s in segments), 2)
        if not MIN_MONTAGE_SECONDS <= total <= MAX_MONTAGE_SECONDS:
            continue
        path = " → ".join(f"Ep {s['episode']}" for s in segments)
        montages.append({
            "montage_id": f"M{len(montages) + 1}",
            "segments": segments,
            "total_duration": total,
            "score": round(sum(p["final_score"] for p, _ in picks) / 3, 2),
            "why_it_works": (
                f"Tells the show's arc in three beats ({path}), using the strongest closing "
                "line available from the start, middle and end of the series."
            ),
            "quality_warning": (
                f"Experimental. The last segment comes from Ep {segments[-1]['episode']}, so check how "
                "much it spoils and whether the beats read as one story when cut together."
            ),
        })
    return montages
