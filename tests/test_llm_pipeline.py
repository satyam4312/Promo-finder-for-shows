import csv
import io
import json
from types import SimpleNamespace

import pytest

from promorank.llm_evaluator import (
    GROQ_DEFAULT_BASE_URL,
    LLMConfigurationError,
    OpenAICompatibleEvaluator,
    _retry_after_seconds,
    resolve_settings,
)
from promorank import pipeline
from promorank.parser import parse_episode_text
from promorank.pipeline import results_to_csv, run_pipeline
from promorank.quality import report_to_markdown
from promorank.train_ranker import FEATURES


def _episode(number: int):
    lines = [
        f"EPISODE NUMBER: Episode {number}",
        "--- SCRIPT START ---",
        "SCENE 01",
        "Scene heading: INT. HALL – NIGHT",
        "Scene start: 00:00.00",
        "Scene end: 02:00.00",
    ]
    speakers = ["WOMAN (Furious)", "MAN (Cold)"]
    quotes = ['"You betrayed me! Why?"', '"Because the truth would ruin you."',
              '"What truth? Tell me now!"', '"Your father is alive, and he wants revenge."']
    for i in range(12):
        start = i * 10
        lines += ["", f"**{speakers[i % 2]}:**",
                  f"Start: (0{start // 60}:{start % 60:02d}.00) | End: (0{(start + 4) // 60}:{(start + 4) % 60:02d}.00)",
                  quotes[i % 4]]
    return parse_episode_text("\n".join(lines) + "\n", f"episode-{number:03d}.txt")


class FakeClient:
    """Mimics client.chat.completions.create and answers each LLM stage.

    Scores every candidate 4/5. ``fail_batches`` counts scoring calls only.
    """

    def __init__(self, fail_batches=(), beats=None):
        self.calls = 0
        self.scoring_calls = 0
        self.prompts = []
        self.fail_batches = set(fail_batches)
        self.beats = beats or []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @staticmethod
    def _reply(content):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][1]["content"]
        self.prompts.append(prompt)
        if "Episode transcript:\n" in prompt:
            return self._reply(json.dumps({"beats": self.beats}))
        if "Pairs:\n" in prompt:
            payload = json.loads(prompt.split("Pairs:\n", 1)[1])
            return self._reply(json.dumps({"comparisons": [{"pair_id": p["pair_id"], "winner": "A"} for p in payload]}))
        self.scoring_calls += 1
        if self.scoring_calls in self.fail_batches:
            return self._reply("not json")
        payload = json.loads(prompt.split("Candidates:\n", 1)[1])
        body = {"evaluations": [{
            "candidate_id": item["candidate_id"], "hook_strength": 4, "cliffhanger_strength": 4,
            "standalone_clarity": 4, "emotional_impact": 4, "why_it_works": "Strong reveal.", "weakness": "Needs context.",
        } for item in payload]}
        return self._reply(json.dumps(body))


def test_resolve_settings_never_mixes_providers():
    key, url, model = resolve_settings(env={"GROQ_API_KEY": "g", "GROQ_MODEL": "m"})
    assert (key, url, model) == ("g", GROQ_DEFAULT_BASE_URL, "m")
    key, url, model = resolve_settings(env={"OPENAI_API_KEY": "o", "OPENAI_MODEL": "x", "GROQ_BASE_URL": "http://groq"})
    assert (key, url, model) == ("o", None, "x")
    with pytest.raises(LLMConfigurationError):
        resolve_settings(env={})


def test_llm_scores_cover_every_episode_and_rank_first():
    episodes = [_episode(n) for n in (1, 2, 3, 4)]
    client = FakeClient()
    results = run_pipeline(episodes, max_results=8, max_per_episode=2, use_llm=True,
                           evaluator=OpenAICompatibleEvaluator(client=client))
    assert results["llm_used"] and results["model_mode"] == "hybrid_llm"
    promos = results["ranked_promos"]
    assert {p["episode"] for p in promos} == {1, 2, 3, 4}
    assert all(p["evaluation_source"] == "hybrid_llm" for p in promos)
    assert all(p["llm_hook"] == 4 for p in promos)
    assert results["llm_error"] is None


def test_partial_llm_failure_is_reported():
    episodes = [_episode(n) for n in (1, 2, 3)]
    results = run_pipeline(episodes, max_results=6, use_llm=True,
                           evaluator=OpenAICompatibleEvaluator(client=FakeClient(fail_batches={1})))
    assert results["llm_used"]
    assert "not valid JSON" in results["llm_error"]
    # LLM-evaluated candidates come first, heuristic fill-ins after.
    sources = [p["evaluation_source"] for p in results["ranked_promos"]]
    assert sources == sorted(sources, key=lambda s: s != "hybrid_llm")


def test_failed_llm_falls_back_to_pure_heuristic():
    class Broken:
        def evaluate(self, *args, **kwargs):
            raise RuntimeError("provider down")

    results = run_pipeline([_episode(1)], max_results=2, use_llm=True, evaluator=Broken())
    assert not results["llm_used"]
    assert "provider down" in results["llm_error"]
    assert all(p["final_score"] == p["heuristic_score"] for p in results["ranked_promos"])


def test_csv_has_every_training_feature():
    results = run_pipeline([_episode(1)], max_results=2)
    header = next(csv.reader(io.StringIO(results_to_csv(results))))
    assert set(FEATURES) <= set(header)


def test_beats_chosen_by_line_id_become_python_built_windows():
    # L1 starts at 0s and L5 ends at 44s. 44s earns the top duration score, so it ties the best windows.
    beats = [{"hook_line": "L1", "cliffhanger_line": "L5", "premise_value": 5, "spoiler_risk": 1, "reason": "Sells it."}]
    results = run_pipeline([_episode(1)], max_results=2, use_llm=True,
                           evaluator=OpenAICompatibleEvaluator(client=FakeClient(beats=beats)))
    from_beat = [p for p in results["ranked_promos"] if p["candidate_source"] == "llm_beat"]
    assert len(from_beat) == 1
    assert (from_beat[0]["start"], from_beat[0]["end"]) == (0.0, 44.0)
    assert from_beat[0]["premise_value"] == 5
    assert results["run_metadata"]["llm_stats"]["beats_usable"] == 1


def test_invented_line_ids_are_rejected_and_reported():
    beats = [{"hook_line": "L999", "cliffhanger_line": "L4"}, {"hook_line": "L4", "cliffhanger_line": "L1"}]
    results = run_pipeline([_episode(1)], max_results=2, use_llm=True,
                           evaluator=OpenAICompatibleEvaluator(client=FakeClient(beats=beats)))
    assert results["run_metadata"]["llm_stats"]["beats_proposed"] == 0
    assert "unknown or reversed line IDs" in results["llm_error"]


def test_judge_sees_next_line_marked_as_not_in_clip():
    client = FakeClient()
    run_pipeline([_episode(1)], max_results=2, use_llm=True, evaluator=OpenAICompatibleEvaluator(client=client))
    scoring = next(p for p in client.prompts if "Candidates:\n" in p)
    assert all("next_line_not_in_clip" in item for item in json.loads(scoring.split("Candidates:\n", 1)[1]))


def test_llm_score_replaces_the_heuristic_and_pairwise_only_nudges_it():
    results = run_pipeline([_episode(n) for n in (1, 2, 3)], max_results=3, use_llm=True,
                           evaluator=OpenAICompatibleEvaluator(client=FakeClient()))
    for promo in results["ranked_promos"]:
        assert promo["pairwise_games"] > 0
        expected = promo["llm_score"] + 3.0 * promo["pairwise_net"] / promo["pairwise_games"]
        assert promo["final_score"] == pytest.approx(expected, abs=0.01)


def test_losing_the_pairwise_round_cannot_drop_a_promo_behind_uncompared_ones(monkeypatch):
    monkeypatch.setattr(pipeline, "PAIRWISE_TOP", 2)

    def candidate(name, score):
        return SimpleNamespace(candidate_id=name, final_score=score, pairwise_net=0, pairwise_games=0)

    winner, loser, uncompared = candidate("w", 80.0), candidate("l", 79.0), candidate("u", 78.9)

    class Evaluator:
        def compare_pairs(self, top, transcripts, context=None, progress=None):
            return {"w": 1, "l": -1}, {"w": 1, "l": 1}, []

    ranked = pipeline._apply_pairwise(Evaluator(), [winner, loser, uncompared], {}, {}, None, [])
    assert ranked == [winner, loser, uncompared]
    assert (winner.final_score, loser.final_score, uncompared.final_score) == (83.0, 78.9, 78.9)


def test_identical_requests_are_answered_from_the_cache(tmp_path):
    episodes = [_episode(1)]
    first_client, second_client = FakeClient(), FakeClient()
    first = OpenAICompatibleEvaluator(client=first_client, cache_dir=tmp_path)
    second = OpenAICompatibleEvaluator(client=second_client, cache_dir=tmp_path)
    run_pipeline(episodes, max_results=2, use_llm=True, evaluator=first)
    run_pipeline(episodes, max_results=2, use_llm=True, evaluator=second)
    assert first_client.calls > 0
    assert second_client.calls == 0
    assert second.describe()["cache_hits"] == first_client.calls


class Fake429(Exception):
    status_code = 429


DAILY_LIMIT = Fake429(
    "Rate limit reached for model `m` in organization `org_x` service tier `on_demand` on tokens per day (TPD): "
    "Limit 200000, Used 199000, Requested 3000. Please try again in 12m30.5s."
)


class ThrottledClient(FakeClient):
    """Raises the given error for the calls listed in ``throttle_calls``, by overall call number."""

    def __init__(self, error, throttle_calls):
        super().__init__()
        self.error = error
        self.throttle_calls = set(throttle_calls)

    def create(self, **kwargs):
        if self.calls + 1 in self.throttle_calls:
            self.calls += 1
            raise self.error
        return super().create(**kwargs)


def test_retry_wait_is_read_from_the_provider_message():
    assert _retry_after_seconds(DAILY_LIMIT) == pytest.approx(750.5)
    assert _retry_after_seconds(Fake429("Please try again in 450ms.")) == pytest.approx(0.45)
    assert _retry_after_seconds(Fake429("Please try again in 1h2m3s.")) == pytest.approx(3723.0)
    assert _retry_after_seconds(Fake429("no hint")) is None


def test_short_rate_limit_is_waited_out_and_the_run_completes():
    waits = []
    client = ThrottledClient(Fake429("Please try again in 7.5s."), {2})
    results = run_pipeline([_episode(1)], max_results=2, use_llm=True,
                           evaluator=OpenAICompatibleEvaluator(client=client, sleep=waits.append))
    assert waits == [8.5]
    assert results["model_mode"] == "hybrid_llm"
    assert results["run_metadata"]["llm_complete"] is True
    assert results["llm_error"] is None


def test_daily_limit_stops_calling_marks_the_run_partial_and_resumes_from_cache(tmp_path):
    episodes = [_episode(n) for n in (1, 2, 3)]
    # Call 1-3 find beats, call 4 scores the first batch, call 5 hits the daily limit.
    limited = ThrottledClient(DAILY_LIMIT, range(5, 500))
    waits = []
    partial = run_pipeline(episodes, max_results=6, use_llm=True,
                           evaluator=OpenAICompatibleEvaluator(client=limited, cache_dir=tmp_path, sleep=waits.append))
    assert limited.calls == 5  # No request is made after the limit is known.
    assert waits == []
    assert partial["model_mode"] == "hybrid_llm_partial"
    assert partial["run_metadata"]["llm_complete"] is False
    assert "tokens per day" in partial["llm_error"] and "org_x" not in partial["llm_error"]
    assert all(p["pairwise_net"] == 0 for p in partial["ranked_promos"])
    assert "INCOMPLETE LLM RUN" in report_to_markdown(partial)

    healthy = FakeClient()
    resumed_evaluator = OpenAICompatibleEvaluator(client=healthy, cache_dir=tmp_path)
    resumed = run_pipeline(episodes, max_results=6, use_llm=True, evaluator=resumed_evaluator)
    assert resumed["model_mode"] == "hybrid_llm"
    assert resumed_evaluator.cache_hits == 4
    assert not any("Episode transcript:\n" in prompt for prompt in healthy.prompts)


def test_run_metadata_records_the_model_and_never_the_key():
    evaluator = OpenAICompatibleEvaluator(client=FakeClient(), api_key="secret-key", model="my-model",
                                          base_url="https://example.test/v1")
    results = run_pipeline([_episode(1)], max_results=2, use_llm=True, evaluator=evaluator)
    llm = results["run_metadata"]["llm"]
    assert llm["model"] == "my-model" and llm["provider_host"] == "example.test" and llm["temperature"] == 0.0
    assert "secret-key" not in json.dumps(results, default=str)
