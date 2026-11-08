"""Per-run audit of the output: where the tool's picks break or look doubtful.

These checks read the finished promo rows and the parsed script, without using the
scoring code's own view of what is inside a clip, so they can catch its mistakes.
"""
from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .parser import Episode, parse_timestamp
from .scoring import Candidate

TOLERANCE = 0.05
GOLD_MATCH_FRACTION = 0.5


def _normalize(text: str) -> str:
    return text.replace("\n", " ").strip()[:260]


def _inside_strictly(t: float, start: float, end: float) -> bool:
    return start + TOLERANCE < t < end - TOLERANCE


def _quoted_line_status(promo: dict, text: str, episode: Episode) -> str:
    """'inside', 'outside', or 'unverified' when the quoted text matches no script line."""
    if not text:
        return "unverified"
    matches = [e for e in episode.events if _normalize(e.text) == _normalize(text)]
    if not matches:
        return "unverified"
    inside = any(e.start >= promo["start"] - TOLERANCE and e.end <= promo["end"] + TOLERANCE for e in matches)
    return "inside" if inside else "outside"


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    try:
        return round(statistics.correlation(xs, ys), 2)
    except statistics.StatisticsError:  # A constant column has no defined correlation.
        return None


def _shared(a: float, b: float, c: float, d: float) -> float:
    shorter = min(b - a, d - c)
    return max(0.0, min(b, d) - max(a, c)) / shorter if shorter > 0 else 0.0


def load_gold(path: str | Path) -> list[dict[str, Any]]:
    """Hand-picked promos: CSV with episode,start,end,note. Times in seconds or MM:SS.ss."""

    def seconds(value: str) -> float:
        return parse_timestamp(value) if ":" in value else float(value)

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [
            {
                "episode": int(row["episode"]),
                "start": seconds(row["start"]),
                "end": seconds(row["end"]),
                "note": (row.get("note") or "").strip(),
            }
            for row in csv.DictReader(handle)
        ]


def _check_gold(
    gold: Iterable[dict[str, Any]],
    promos: list[dict[str, Any]],
    candidates: list[Candidate],
) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    rows = []
    for item in gold:
        match = next(
            (
                p for p in promos
                if p["episode"] == item["episode"]
                and _shared(item["start"], item["end"], p["start"], p["end"]) >= GOLD_MATCH_FRACTION
            ),
            None,
        )
        nearest_rank = next(
            (
                position for position, c in enumerate(ordered, 1)
                if c.episode == item["episode"]
                and _shared(item["start"], item["end"], c.start, c.end) >= GOLD_MATCH_FRACTION
            ),
            None,
        )
        rows.append({
            **item,
            "picked": match is not None,
            "picked_rank": match["rank"] if match else None,
            "best_candidate_rank": nearest_rank,
        })
    hits = sum(r["picked"] for r in rows)
    return {"total": len(rows), "picked": hits, "recall": round(hits / len(rows), 2) if rows else None, "items": rows}


def build_quality_report(
    promos: list[dict[str, Any]],
    episodes: list[Episode],
    candidates: list[Candidate],
    gold: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_number = {e.number: e for e in episodes}
    counts: Counter = Counter()
    grouped: dict[int, list[dict]] = defaultdict(list)
    for promo in promos:
        episode = by_number[promo["episode"]]
        grouped[promo["episode"]].append(promo)
        counts[f"hook_line_{_quoted_line_status(promo, promo['hook'], episode)}"] += 1
        counts[f"cliffhanger_line_{_quoted_line_status(promo, promo['cliffhanger'], episode)}"] += 1
        for edge, t in (("start", promo["start"]), ("end", promo["end"])):
            for kind, name in (("camera", "shot"), ("action", "action"), ("dialogue", "line")):
                if any(e.event_type == kind and _inside_strictly(t, e.start, e.end) for e in episode.events):
                    counts[f"{edge}s_mid_{name}"] += 1
        if abs(promo["end"] - episode.duration) <= 0.5:
            counts["ends_at_episode_end"] += 1

    overlapping = nested = over_limit = 0
    for group in grouped.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                shared = _shared(a["start"], a["end"], b["start"], b["end"])
                overlapping += shared > 0
                over_limit += shared > 0.3
                nested += shared >= 0.95

    scores = [p["final_score"] for p in promos]
    top = scores[:10]
    per_episode = Counter(len(g) for g in grouped.values())
    judged = [p for p in promos if p.get("llm_score") is not None]

    agreement = None
    disagreements: list[dict[str, Any]] = []
    if judged:
        pairs = {
            "overall": ("heuristic_score", "llm_score"),
            "hook": ("hook_score", "llm_hook"),
            "cliffhanger": ("cliffhanger_score", "llm_cliffhanger"),
            "standalone": ("standalone_score", "llm_standalone"),
            "emotion": ("emotion_score", "llm_emotion"),
        }
        agreement = {
            name: _correlation([float(p[x]) for p in judged], [float(p[y]) for p in judged])
            for name, (x, y) in pairs.items()
        }
        widest = sorted(judged, key=lambda p: abs(p["llm_score"] - p["heuristic_score"]), reverse=True)[:10]
        disagreements = [{
            "rank": p["rank"], "episode": p["episode"], "time_range": p["time_range"],
            "heuristic_score": p["heuristic_score"], "llm_score": p["llm_score"],
            "llm_weakness": p["quality_warning"],
        } for p in widest]

    hook_values = Counter(p["llm_hook"] for p in judged)
    weakest = sorted(promos, key=lambda p: p["cliffhanger_score"])[:5]

    return {
        "promos_checked": len(promos),
        "line_integrity": {
            "hook_line_outside_clip": counts["hook_line_outside"],
            "cliffhanger_line_outside_clip": counts["cliffhanger_line_outside"],
            "hook_line_unverified": counts["hook_line_unverified"],
            "cliffhanger_line_unverified": counts["cliffhanger_line_unverified"],
        },
        "cuts": {
            "starts_mid_line": counts["starts_mid_line"],
            "ends_mid_line": counts["ends_mid_line"],
            "starts_mid_action": counts["starts_mid_action"],
            "ends_mid_action": counts["ends_mid_action"],
            "starts_mid_shot": counts["starts_mid_shot"],
            "ends_mid_shot": counts["ends_mid_shot"],
        },
        "same_episode_overlap": {
            "overlapping_pairs": overlapping,
            "pairs_over_30_percent_shared": over_limit,
            "nested_pairs": nested,
        },
        "ranking_shape": {
            "ends_at_episode_end": counts["ends_at_episode_end"],
            "episodes_represented": len(grouped),
            "promos_per_episode": {str(k): v for k, v in sorted(per_episode.items())},
            "top_10_score_spread": round(top[0] - top[-1], 2) if len(top) > 1 else None,
            "score_stdev": round(statistics.pstdev(scores), 2) if len(scores) > 1 else None,
            "most_common_llm_hook_share": (
                round(hook_values.most_common(1)[0][1] / len(judged), 2) if judged else None
            ),
        },
        "heuristic_vs_llm_correlation": agreement,
        "largest_disagreements": disagreements,
        "weakest_endings_in_list": [{
            "rank": p["rank"], "episode": p["episode"], "time_range": p["time_range"],
            "cliffhanger_score": p["cliffhanger_score"], "cliffhanger": p["cliffhanger"],
        } for p in weakest],
        "gold_check": _check_gold(gold, promos, candidates) if gold else None,
    }


_MODE_DESCRIPTIONS = {
    "hybrid_llm": "Complete LLM run: the LLM proposed moments and scored the clips.",
    "hybrid_llm_partial": "Incomplete LLM run: it stopped early, so this ranking is not final.",
    "deterministic_baseline": "Rule-based scorer only: no LLM was used.",
}


def _cell(value: Any) -> str:
    """Make a value safe to place inside a Markdown table cell."""
    return "n/a" if value is None else str(value).replace("\n", " ").replace("|", "\\|").strip()


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return lines + [""]


def _pass_fail(count: int) -> str:
    return "Pass" if count == 0 else "Fail"


def _agreement(value: float | None) -> str:
    if value is None:
        return "Cannot be measured (one side has no variation)"
    if value < 0.2:
        return "Almost no agreement"
    if value < 0.4:
        return "Weak agreement"
    if value < 0.7:
        return "Moderate agreement"
    return "Strong agreement"


def _about_run(results: dict[str, Any]) -> list[list[Any]]:
    meta = results["run_metadata"]
    dataset = results["dataset"]
    rows = [
        ["Generated", meta["generated_at"]],
        ["Mode", _MODE_DESCRIPTIONS.get(meta["mode"], meta["mode"])],
        ["Scripts read", f"{dataset['episode_count']} episodes, {dataset['event_count']:,} timestamped blocks"],
        ["Script format warnings", len(dataset.get("parse_warnings", []))],
        ["Valid 20-60 second windows found", f"{results['candidate_count']:,}"],
    ]
    llm = meta.get("llm")
    if llm:
        stats = meta.get("llm_stats") or {}
        rows += [
            ["Model", f"{llm.get('model')} via {llm.get('provider_host')}"],
            ["Model settings", f"Temperature {llm.get('temperature')}, prompt version {llm.get('prompt_version')}"],
            ["Moments proposed by the LLM", f"{stats.get('beats_proposed', 0)}, of which {stats.get('beats_usable', 0)} became valid clips"],
            ["Clips scored by the LLM", results.get("llm_evaluated_count", 0)],
            ["LLM work for this report", (
                f"{llm.get('api_calls')} new calls and {llm.get('cache_hits')} saved replies reused "
                f"({llm.get('tokens_used')} new tokens)"
            )],
        ]
    return rows


def report_to_markdown(results: dict[str, Any]) -> str:
    report = results["quality_report"]
    meta = results["run_metadata"]
    promos = results["ranked_promos"]
    integrity, cuts = report["line_integrity"], report["cuts"]
    overlap, shape = report["same_episode_overlap"], report["ranking_shape"]
    gold = report["gold_check"]
    total = report["promos_checked"]

    failures = (
        sum(integrity.values()) + sum(cuts.values())
        + overlap["pairs_over_30_percent_shared"] + overlap["nested_pairs"]
    )
    if failures:
        verdict = f"**{failures} automatic check failures were found across {total} promos.** Sections 1 and 2 show where."
    else:
        verdict = (
            f"**All {total} promos passed every automatic check.** No hook or ending falls outside its clip, "
            "no cut interrupts a line, an action or a shot, and no two promos from one episode are near-copies."
        )
    if gold:
        measured = f"Against the hand-picked list, the tool found {gold['picked']} of {gold['total']} promos."
    else:
        measured = "No hand-picked list was supplied, so how often the tool finds the best promos is unmeasured."

    lines = ["# PromoRank quality report", "", verdict, ""]
    if meta.get("llm_complete") is False:
        lines += [
            "**INCOMPLETE LLM RUN. Do not use this ranking.** Scored promos rank ahead of unscored ones "
            f"regardless of quality. Reason: {results.get('llm_error')}",
            "",
        ]
    lines += [
        f"These checks show that the clips are technically clean, not that they are good promos. {measured}",
        "",
        "## About this run",
        "",
        *_table(["Item", "Value"], _about_run(results)),
        "## 1. Is each clip technically clean?",
        "",
        "A promo is unusable if it starts or ends mid-sentence, or if the line it relies on is cut off. "
        "Each number below counts the promos with that problem, so 0 is the target.",
        "",
        *_table(["Check", "Promos affected", "Result", "Why it matters"], [
            ["Hook line outside the clip", integrity["hook_line_outside_clip"], _pass_fail(integrity["hook_line_outside_clip"]), "The viewer must see or hear the whole opening beat."],
            ["Cliffhanger line outside the clip", integrity["cliffhanger_line_outside_clip"], _pass_fail(integrity["cliffhanger_line_outside_clip"]), "The viewer must hear the whole closing line."],
            ["Hook text not found in the script", integrity["hook_line_unverified"], _pass_fail(integrity["hook_line_unverified"]), "Every quoted hook must be real script text."],
            ["Cliffhanger text not found in the script", integrity["cliffhanger_line_unverified"], _pass_fail(integrity["cliffhanger_line_unverified"]), "Every quoted ending must be real script text."],
            ["Starts in the middle of a line", cuts["starts_mid_line"], _pass_fail(cuts["starts_mid_line"]), "Opening mid-sentence sounds broken."],
            ["Ends in the middle of a line", cuts["ends_mid_line"], _pass_fail(cuts["ends_mid_line"]), "Ending mid-sentence sounds broken."],
            ["Starts in the middle of an action", cuts["starts_mid_action"], _pass_fail(cuts["starts_mid_action"]), "A slap or fall needs to be seen whole."],
            ["Ends in the middle of an action", cuts["ends_mid_action"], _pass_fail(cuts["ends_mid_action"]), "A slap or fall needs to be seen whole."],
            ["Starts in the middle of a shot", cuts["starts_mid_shot"], _pass_fail(cuts["starts_mid_shot"]), "Cuts look clean on a change of camera shot."],
            ["Ends in the middle of a shot", cuts["ends_mid_shot"], _pass_fail(cuts["ends_mid_shot"]), "Cuts look clean on a change of camera shot."],
        ]),
        "## 2. Are the promos from one episode really different?",
        "",
        "Two picks from one episode should not be the same moment twice. Overlap is measured against the shorter "
        "clip, so a short clip sitting inside a longer one counts as a full overlap.",
        "",
        *_table(["Check", "Pairs", "Result", "What it means"], [
            ["Pairs sharing some time", overlap["overlapping_pairs"], "Allowed", "A small shared stretch is fine, up to 30% of the shorter clip."],
            ["Pairs sharing more than 30%", overlap["pairs_over_30_percent_shared"], _pass_fail(overlap["pairs_over_30_percent_shared"]), "These would be too similar to list twice."],
            ["Nested copies", overlap["nested_pairs"], _pass_fail(overlap["nested_pairs"]), "One clip sits almost entirely inside another."],
        ]),
        "## 3. What does the ranking look like?",
        "",
        "These figures have no pass or fail. They describe how the list is spread and how clearly the scores separate promos.",
        "",
    ]

    spread = ", ".join(
        f"{count} with {size} promo{'' if size == '1' else 's'}" for size, count in shape["promos_per_episode"].items()
    )
    shape_rows = [
        ["Episodes represented", shape["episodes_represented"], "Every episode gets at least one promo where a valid clip exists."],
        ["Promos per episode", f"Episodes: {spread}", "The strongest episodes can earn more than one place."],
        ["Promos ending at an episode's end", shape["ends_at_episode_end"], "Episodes are written to end on a cliffhanger, so these are often strong."],
        ["Score gap across the top 10", shape["top_10_score_spread"], "A larger gap means the order at the top is less of a coin toss."],
        ["Score spread across the list (standard deviation)", shape["score_stdev"], "Higher means the scores separate promos more clearly."],
    ]
    judged = [p for p in promos if p.get("llm_score") is not None]
    if judged:
        shape_rows.append([
            "Most common LLM hook rating",
            f"{shape['most_common_llm_hook_share']:.0%} of promos share it",
            "A high share means the LLM is giving many hooks the same rating.",
        ])
        compared = sum(1 for p in promos if p.get("pairwise_games"))
        shape_rows.append(["Promos compared head to head", compared, "These get a small tiebreak of up to 3 points either way."])
        unscored = [p["rank"] for p in promos if p.get("llm_score") is None]
        if unscored:
            shape_rows.append([
                "Ranks not scored by the LLM",
                f"{unscored[0]}-{unscored[-1]}",
                "They are placed last whatever their score, because the two scoring methods are not comparable.",
            ])
    lines += _table(["Measure", "Value", "How to read it"], shape_rows)

    correlation = report["heuristic_vs_llm_correlation"]
    if correlation:
        lines += [
            "## 4. Do the rule-based scorer and the LLM agree?",
            "",
            "Correlation runs from -1 to 1, where 1 means the two methods rate promos the same way and 0 means no "
            "relationship. It covers only the promos in the final list, which makes agreement look lower than it "
            "would across all clips. Low agreement does not show which method is right.",
            "",
            *_table(["Dimension", "Correlation", "Reading"], [
                [name.capitalize(), value, _agreement(value)] for name, value in correlation.items()
            ]),
            "## 5. Where do they disagree most?",
            "",
            "These promos have the widest gap between the two scores, so they are the best ones to review by hand. "
            "Scores are out of 100, and the last column is the LLM's stated concern.",
            "",
            *_table(["Rank", "Ep", "Segment", "Rule-based", "LLM", "LLM's concern"], [
                [d["rank"], d["episode"], d["time_range"], f"{d['heuristic_score']:.1f}", f"{d['llm_score']:.1f}", d["llm_weakness"]]
                for d in report["largest_disagreements"]
            ]),
        ]

    number = 6 if correlation else 4
    lines += [
        f"## {number}. Which endings look weakest?",
        "",
        "These have the lowest ending scores from the rule-based scorer, out of 5. That scorer relies on keywords "
        "and punctuation, so a quiet but powerful line can land here. Treat this as a list to check, not a list of mistakes.",
        "",
        *_table(["Rank", "Ep", "Segment", "Ending score", "Final line"], [
            [w["rank"], w["episode"], w["time_range"], w["cliffhanger_score"], w["cliffhanger"]]
            for w in report["weakest_endings_in_list"]
        ]),
        f"## {number + 1}. Did it find the promos a person would choose?",
        "",
    ]
    if gold:
        lines += [
            f"The tool picked {gold['picked']} of {gold['total']} hand-picked promos, a recall of {gold['recall']}. "
            "A pick counts as found when it shares at least half of the shorter clip with the hand-picked one.",
            "",
            *_table(["Ep", "Hand-picked time (s)", "Result", "Best matching window", "Note"], [
                [
                    item["episode"],
                    f"{item['start']:.1f}-{item['end']:.1f}",
                    f"Picked at #{item['picked_rank']}" if item["picked"] else "Missed",
                    f"Ranks {item['best_candidate_rank']}" if item["best_candidate_rank"] else "No valid window matches it",
                    item["note"],
                ]
                for item in gold["items"]
            ]),
        ]
    else:
        lines += [
            "Not measured. Pass a CSV of hand-picked promos with `--gold` (columns: episode, start, end, note) "
            "to see how many the tool finds and where the ones it missed rank.",
            "",
        ]
    lines += [
        "## What this report cannot tell you",
        "",
        "- Whether viewers will actually watch or click. Only real performance data can show that.",
        "- How the acting, pacing or visuals play. The tool reads screenplays, not finished video.",
        "- Whether screenplay timecodes match the final edit. Check timestamps against the video before publishing.",
        "",
    ]
    return "\n".join(lines)
