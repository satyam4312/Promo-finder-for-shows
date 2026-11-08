"""End-to-end screenplay-to-promo pipeline."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .curation import build_montages, select_post_first
from .parser import Episode, load_zip_bytes
from .quality import build_quality_report, report_to_markdown
from .scoring import (
    Candidate,
    clip_transcript,
    diversify,
    generate_candidates,
    line_after_cut,
    timestamp_range,
    window_for_lines,
)

PIPELINE_VERSION = "2.0"
CSV_FIELDS = [
    "rank", "post_first_rank", "candidate_id", "episode", "segments", "time_range", "start", "end", "duration",
    "final_score", "evaluation_source", "candidate_source", "pairwise_net", "pairwise_games", "premise_adjustment", "heuristic_score",
    "hook_score", "cliffhanger_score", "standalone_score", "emotion_score", "duration_score",
    "llm_score", "llm_hook", "llm_cliffhanger", "llm_standalone", "llm_emotion",
    "premise_value", "spoiler_risk", "hook_at", "hook_type", "at_episode_end",
    "hook", "cliffhanger", "why_it_works", "quality_warning", "scene_numbers", "dialogue_excerpt",
]
# Head-to-head rounds only reorder the top of the list, where 1-5 scores cannot separate promos.
PAIRWISE_TOP = 30
# Most a promo can gain or lose, at a 100% or 0% win rate. Each promo only meets its neighbours,
# so this has to stay a tiebreak rather than a score that lets it leap over promos it never met.
PAIRWISE_MAX_POINTS = 3.0
# Score change per point of premise value above or below the list's average beat.
PREMISE_POINTS = 3.0
# The LLM pool may hold alternatives that share more than the final list allows.
POOL_MAX_SHARED = 0.7


def _windows_from_beats(
    beats: list[dict[str, Any]],
    episodes_by_number: dict[int, Episode],
    candidates: list[Candidate],
) -> list[Candidate]:
    """Validate LLM-chosen line pairs into windows. Beats that cannot make a clean clip are dropped."""
    index = {c.candidate_id: c for c in candidates}
    found: dict[str, Candidate] = {}
    for beat in beats:
        window = window_for_lines(episodes_by_number[beat["episode"]], beat["hook_event"], beat["cliff_event"])
        if window is None or window.candidate_id in found:
            continue
        candidate = index.get(window.candidate_id)
        if candidate is None:
            candidate = window
            index[window.candidate_id] = window
            candidates.append(window)
        candidate.candidate_source = "llm_beat"
        candidate.premise_value = beat["premise_value"]
        candidate.spoiler_risk = beat["spoiler_risk"]
        candidate.beat_reason = beat["reason"]
        found[window.candidate_id] = candidate
    return list(found.values())


def _build_llm_pool(
    evaluator,
    episodes: list[Episode],
    candidates: list[Candidate],
    pool_size: int,
    limit: int | None,
    max_per_episode: int,
    progress: Callable[[int, int], None] | None,
    failures: list[str],
    stats: dict[str, int],
) -> list[Candidate]:
    """LLM-found beats first, plus heuristic picks so every episode keeps an alternative."""
    per_episode = max_per_episode + 1
    beat_windows: list[Candidate] = []
    find_beats = getattr(evaluator, "find_beats", None)
    if find_beats is not None:
        beats, beat_failures = find_beats(episodes, progress=progress)
        failures.extend(beat_failures)
        beat_windows = _windows_from_beats(beats, {e.number: e for e in episodes}, candidates)
        stats["beats_proposed"] = len(beats)
        stats["beats_usable"] = len(beat_windows)

    beat_pool = diversify(beat_windows, max_results=pool_size, max_per_episode=per_episode, max_shared=POOL_MAX_SHARED)
    beat_ids = {c.candidate_id for c in beat_pool}
    fill_count = max(pool_size - len(beat_pool), len(episodes)) if beat_pool else pool_size
    fill = diversify(
        [c for c in candidates if c.candidate_id not in beat_ids],
        max_results=fill_count,
        max_per_episode=per_episode,
        max_shared=POOL_MAX_SHARED,
        preselected=beat_pool,
    )
    pool = beat_pool + fill
    return pool[:limit] if limit else pool


def _apply_pairwise(
    evaluator,
    ranked: list[Candidate],
    transcripts: dict[str, str],
    context: dict[str, dict[str, str]],
    progress: Callable[[int, int], None] | None,
    failures: list[str],
) -> list[Candidate]:
    compare = getattr(evaluator, "compare_pairs", None)
    top = ranked[:PAIRWISE_TOP]
    if compare is None or len(top) < 2:
        return ranked
    net, games, pair_failures = compare(top, transcripts, context=context, progress=progress)
    failures.extend(pair_failures)
    if getattr(evaluator, "quota_message", None):
        return ranked  # A half-finished round would only reward the promos compared so far.
    # Promos below the cut were never compared, so losing a round must not put a promo behind them.
    floor = max((c.final_score for c in ranked[PAIRWISE_TOP:]), default=float("-inf"))
    for candidate in top:
        candidate.pairwise_net = net.get(candidate.candidate_id, 0)
        candidate.pairwise_games = games.get(candidate.candidate_id, 0)
        if candidate.pairwise_games:
            win_rate_margin = candidate.pairwise_net / candidate.pairwise_games
            candidate.final_score = round(max(floor, candidate.final_score + PAIRWISE_MAX_POINTS * win_rate_margin), 2)
    return sorted(ranked, key=lambda c: c.final_score, reverse=True)


def _apply_premise(ranked: list[Candidate]) -> list[Candidate]:
    """Reward beats that sell the show. Promos without a premise rating are left unchanged.

    The adjustment is relative to the list's average, so LLM beats gain no blanket edge
    over unrated promos. It runs after the head-to-head round so that the promos compared
    there, and their cached replies, stay the same.
    """
    rated = [c.premise_value for c in ranked if c.premise_value is not None]
    if not rated:
        return ranked
    average = sum(rated) / len(rated)
    for candidate in ranked:
        if candidate.premise_value is not None:
            candidate.premise_adjustment = round(PREMISE_POINTS * (candidate.premise_value - average), 2)
            candidate.final_score = round(candidate.final_score + candidate.premise_adjustment, 2)
    return sorted(ranked, key=lambda c: c.final_score, reverse=True)


def run_pipeline(
    episodes: list[Episode],
    max_results: int = 100,
    max_per_episode: int = 3,
    use_llm: bool = False,
    llm_candidates: int | None = None,
    evaluator=None,
    progress: Callable[[int, int], None] | None = None,
    llm_options: dict[str, Any] | None = None,
    gold: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate, score and rank promo candidates.

    With ``use_llm`` the LLM reads each episode and names its hook and cliffhanger lines,
    Python builds and validates those windows, the LLM judges them, and the top of the
    list is settled by head-to-head comparison. Without it the heuristic ranks everything.
    """
    if max_results < 1 or max_per_episode < 1:
        raise ValueError("max_results and max_per_episode must be at least 1.")
    candidates = generate_candidates(episodes)
    failures: list[str] = []
    llm_error = None
    report = None
    stats = {"beats_proposed": 0, "beats_usable": 0, "pool_size": 0}
    ranked: list[Candidate]
    by_episode = {e.number: e for e in episodes}
    premise = episodes[0].logline if episodes else ""
    transcripts: dict[str, str] = {}
    context: dict[str, dict[str, str]] = {}

    if use_llm and candidates:
        pool_size = max(max_results, llm_candidates or int(max_results * 1.5))
        limit = pool_size if llm_candidates else None
        try:
            if evaluator is None:
                from .llm_evaluator import OpenAICompatibleEvaluator
                evaluator = OpenAICompatibleEvaluator(**(llm_options or {}))
            pool = _build_llm_pool(
                evaluator, episodes, candidates, pool_size, limit, max_per_episode, progress, failures, stats
            )
            stats["pool_size"] = len(pool)
            transcripts = {c.candidate_id: clip_transcript(by_episode[c.episode], c.start, c.end) for c in pool}
            context = {
                c.candidate_id: {
                    "next_line_not_in_clip": line_after_cut(by_episode[c.episode], c.end),
                    "series_premise": premise,
                }
                for c in pool
            }
            report = evaluator.evaluate(pool, transcripts, progress=progress, context=context)
            failures.extend(report.failures)
            if report.summary or failures:
                llm_error = report.summary or "; ".join(failures[:3])
        except Exception as exc:  # Keep the deterministic pipeline usable when provider/API is unavailable.
            llm_error = f"LLM evaluation unavailable: {type(exc).__name__}: {exc}"

    llm_used = bool(report and report.evaluated)
    if llm_used:
        # LLM scores and heuristic-only scores are not on the same footing,
        # so rank LLM-evaluated candidates first and only then fill any gaps.
        # Integer 1-5 scores tie often; a stable sort lets beats the LLM chose win those ties.
        evaluated = sorted(
            (c for c in candidates if c.llm_evaluation),
            key=lambda c: c.candidate_source != "llm_beat",
        )
        ranked = diversify(evaluated, max_results=max_results, max_per_episode=max_per_episode)
        try:
            pair_failures: list[str] = []
            ranked = _apply_pairwise(evaluator, ranked, transcripts, context, progress, pair_failures)
            if pair_failures:
                llm_error = "; ".join(filter(None, (llm_error, *pair_failures[:3])))
        except Exception as exc:  # The scored ranking is still valid without the head-to-head round.
            llm_error = "; ".join(filter(None, (llm_error, f"Pairwise round failed: {type(exc).__name__}: {exc}")))
        ranked = _apply_premise(ranked)
        if len(ranked) < max_results:
            rest = [c for c in candidates if not c.llm_evaluation]
            ranked += diversify(rest, max_results=max_results - len(ranked),
                                max_per_episode=max_per_episode, preselected=ranked)
    else:
        for c in candidates:  # A failed LLM run must not leave partially blended scores behind.
            c.final_score, c.llm_evaluation = c.heuristic_score, None
            c.candidate_source, c.premise_value, c.spoiler_risk, c.beat_reason = "heuristic", None, None, ""
        ranked = diversify(candidates, max_results=max_results, max_per_episode=max_per_episode)

    promos = [_promo_row(c, i + 1, llm_used) for i, c in enumerate(ranked)]
    post_first = select_post_first(promos, episodes)
    montages = build_montages(promos, episodes)
    stopped_early = getattr(evaluator, "quota_message", None)
    llm_complete = llm_used and report.evaluated == report.requested and not stopped_early
    mode = "deterministic_baseline" if not llm_used else "hybrid_llm" if llm_complete else "hybrid_llm_partial"
    if stopped_early:
        llm_error = "; ".join(dict.fromkeys(filter(None, (stopped_early, llm_error))))
    describe = getattr(evaluator, "describe", None) if llm_used else None
    return {
        "project": "PromoRank AI",
        "model_mode": mode,
        "llm_used": llm_used,
        "llm_error": llm_error,
        "llm_evaluated_count": report.evaluated if report else 0,
        "run_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pipeline_version": PIPELINE_VERSION,
            "mode": mode,
            "llm_complete": llm_complete if use_llm else None,
            "llm": describe() if describe else None,
            "llm_stats": stats if use_llm else None,
            "settings": {
                "max_results": max_results,
                "max_per_episode": max_per_episode,
                "llm_candidates": llm_candidates,
                "pairwise_top": PAIRWISE_TOP if llm_used else 0,
                "premise_points": PREMISE_POINTS if llm_used else 0,
            },
        },
        "dataset": {
            "episode_count": len(episodes),
            "scene_count": sum(len(e.scenes) for e in episodes),
            "event_count": sum(len(e.events) for e in episodes),
            "dialogue_block_count": sum(sum(1 for ev in e.events if ev.event_type == "dialogue") for e in episodes),
            "parse_warnings": [warning for e in episodes for warning in e.parse_warnings],
        },
        "candidate_count": len(candidates),
        "ranked_promos": promos,
        "post_first": post_first,
        "montages": montages,
        "quality_report": build_quality_report(promos, episodes, candidates, gold),
    }


def _promo_row(candidate: Candidate, rank: int, llm_used: bool) -> dict[str, Any]:
    llm = candidate.llm_evaluation or {}
    if llm:
        why = llm.get("why_it_works") or candidate.beat_reason or candidate.rationale
        warning = llm.get("weakness") or "Review narrative context manually."
    elif llm_used:
        why = candidate.rationale
        warning = "Not evaluated by the LLM (batch failed or pool exhausted); heuristic score only."
    else:
        why = candidate.rationale
        warning = (
            "Low-confidence heuristic result; editor review is especially important."
            if candidate.final_score < 65 or candidate.cliffhanger_score < 2.5
            else "Heuristic-only result; review narrative context manually."
        )
    return {
        **candidate.to_dict(),
        # A single clip is a one-item list, so montages use the same shape.
        "segments": [{"episode": candidate.episode, "start": candidate.start, "end": candidate.end}],
        "time_range": timestamp_range(candidate),
        "rank": rank,
        "post_first_rank": None,
        "evaluation_source": "hybrid_llm" if llm else "heuristic",
        "llm_score": llm.get("llm_score"),
        "llm_hook": llm.get("hook_strength"),
        "llm_cliffhanger": llm.get("cliffhanger_strength"),
        "llm_standalone": llm.get("standalone_clarity"),
        "llm_emotion": llm.get("emotional_impact"),
        "why_it_works": why,
        "quality_warning": warning,
    }


def run_zip_pipeline(zip_bytes: bytes, **kwargs) -> dict[str, Any]:
    return run_pipeline(load_zip_bytes(zip_bytes), **kwargs)


def results_to_csv(results: dict[str, Any]) -> str:
    """Ranked promos as CSV text. Includes every column train_ranker.py needs."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in results.get("ranked_promos", []):
        writer.writerow({
            **row,
            "scene_numbers": " ".join(str(n) for n in row.get("scene_numbers") or []),
            "segments": json.dumps(row.get("segments") or []),
        })
    return buffer.getvalue()


def save_results(results: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "promorank_results.json"
    csv_path = output / "ranked_promos.csv"
    report_path = output / "quality_report.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        handle.write(results_to_csv(results))
    report_path.write_text(report_to_markdown(results), encoding="utf-8")
    return json_path, csv_path, report_path
