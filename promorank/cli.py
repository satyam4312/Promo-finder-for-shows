from __future__ import annotations

import argparse

from .parser import load_zip_path
from .pipeline import run_pipeline, save_results
from .quality import load_gold


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and rank microdrama promo candidates from screenplay ZIP files.")
    parser.add_argument("--input", required=True, help="Path to ZIP containing episode .txt files")
    parser.add_argument("--output", default="raw_outputs", help="Directory for JSON, CSV and quality report")
    parser.add_argument("--max-results", type=_positive_int, default=100)
    parser.add_argument("--max-per-episode", type=_positive_int, default=3)
    parser.add_argument("--llm", action="store_true", help="Enable optional OpenAI-compatible LLM stages")
    parser.add_argument("--llm-candidates", type=_positive_int, default=None,
                        help="Cap on how many windows the LLM scores (default: 1.5x --max-results, plus one per episode)")
    parser.add_argument("--model", default=None, help="Model ID; overrides the environment")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint; overrides the environment")
    parser.add_argument("--cache-dir", default=".llm_cache", help="Where LLM responses are cached for free, repeatable reruns")
    parser.add_argument("--no-cache", action="store_true", help="Always call the provider")
    parser.add_argument("--gold", default=None, help="CSV of hand-picked promos (episode,start,end,note) to check recall against")
    args = parser.parse_args()

    def progress(done: int, total: int) -> None:
        print(f"\rLLM step: {done}/{total}", end="" if done < total else "\n", flush=True)

    episodes = load_zip_path(args.input)
    llm_options = {
        "model": args.model,
        "base_url": args.base_url,
        "cache_dir": None if args.no_cache else args.cache_dir,
        "on_wait": lambda seconds: print(f"\nRate limited; waiting {seconds:.0f}s...", flush=True),
    }
    results = run_pipeline(
        episodes,
        max_results=args.max_results,
        max_per_episode=args.max_per_episode,
        use_llm=args.llm,
        llm_candidates=args.llm_candidates,
        progress=progress if args.llm else None,
        llm_options=llm_options,
        gold=load_gold(args.gold) if args.gold else None,
    )
    json_path, csv_path, report_path = save_results(results, args.output)
    for warning in results["dataset"]["parse_warnings"]:
        print(f"PARSE WARNING: {warning}")
    print(f"Parsed episodes: {results['dataset']['episode_count']}")
    print(f"Scenes: {results['dataset']['scene_count']}")
    print(f"Timestamped blocks: {results['dataset']['event_count']}")
    print(f"Candidate windows: {results['candidate_count']}")
    print(f"Scoring mode: {results['model_mode']} (LLM-evaluated: {results['llm_evaluated_count']})")
    print(f"Ranked promos: {len(results['ranked_promos'])}")
    print(f"Post-first tier: {len(results['post_first'])}")
    print(f"Montages: {len(results['montages'])}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Quality report: {report_path}")
    if results.get("llm_error"):
        print(results["llm_error"])
    if results["run_metadata"].get("llm_complete") is False:
        print("INCOMPLETE LLM RUN: do not submit this output. Rerun the same command to continue from the cache.")


if __name__ == "__main__":
    main()
