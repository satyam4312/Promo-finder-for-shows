# PromoRank AI

A prototype that finds and ranks 20–60 second promotional clip candidates from timestamped microdrama screenplays.

## What it does

1. Parses episode metadata, scene boundaries, and timestamped dialogue/action/camera/sound blocks from `.txt` files inside a ZIP, and warns if the number of `**LABEL:**` headers does not match the blocks parsed.
2. Builds 20–60 second windows between shot boundaries (the CAMERA block edges). A window is rejected if a cut would land inside a spoken line or an action, if it opens on "..." or "But", or if nothing happens in its first 3 seconds.
3. Scores only what the viewer fully gets. A line counts only if it plays completely inside the clip.
4. Rejects overlapping picks by measuring shared time against the shorter clip (at most 30%), so nested near-copies cannot both appear.
5. Takes the strongest window from each episode, then fills the list by score, up to a per-episode cap.
6. Suggests a short post-first tier, and builds experimental cross-episode montages.
7. Writes a ranked CSV, complete JSON, and a quality report on every run.

## Important modeling limitation

The supplied pack contains screenplays but no human-labeled examples of successful promos, so this is **not** a supervised model trained on promo labels. It is a transparent heuristic baseline with an optional LLM layer. A later supervised ranker should be trained only after editors label candidate clips.

The LLM never writes timestamps. It names line IDs, and Python turns those into windows and validates them against the screenplay timing.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Upload the screenplay ZIP in the app. The ZIP is processed in memory. Do not commit the confidential screenplay pack or derived raw outputs to a public repository.

## CLI

```bash
python -m promorank.cli --input path/to/screenplays.zip --output raw_outputs
```

| Flag | Purpose |
|---|---|
| `--max-results` | Promos to return (default 100) |
| `--max-per-episode` | Cap per episode (default 3) |
| `--llm` | Turn on the LLM stages |
| `--model`, `--base-url` | Use any OpenAI-compatible model or endpoint; these override the environment |
| `--llm-candidates` | Cap on how many windows the LLM scores |
| `--cache-dir`, `--no-cache` | Control the LLM response cache (default `.llm_cache`) |
| `--gold` | CSV of hand-picked promos to measure recall against |

## Optional LLM stages

Create a file named `.env` in the project root:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_available_model_id
# Optional for Groq or another OpenAI-compatible provider:
OPENAI_BASE_URL=https://api.groq.com/openai/v1
```

For Groq you can instead use `GROQ_API_KEY` and `GROQ_MODEL`. Settings come from one provider only: if `OPENAI_API_KEY` is set, all `OPENAI_*` values are used; otherwise all `GROQ_*` values.

How a run works:

1. **Find beats.** One call per episode sends the numbered transcript, the series premise and the episode logline. The LLM returns hook and cliffhanger line IDs, plus premise value and spoiler risk.
2. **Build windows.** Python starts at a clean cut just before the hook line and ends at a clean cut just after the cliffhanger line. Windows that fail validation are dropped.
3. **Judge.** The LLM scores those windows plus one heuristic alternative per episode. It sees only lines inside the clip, and the next line marked as not in the clip, so it can tell whether the question really stays open.
4. **Compare.** The top 30 are compared head to head against their neighbours, because 1–5 scores bunch together.

For 50 episodes with the defaults this is about 100–110 calls: 50 to find beats, 30–40 to score and about 20 to compare. Calls run at temperature 0 and successful replies are cached, so a rerun of the same input is free and repeatable.

The final score of an LLM-judged promo is the LLM score, plus up to 3 points either way from its head-to-head win rate (a tiebreak within the top 30, which can never push a promo below the ones that were not compared), plus 3 points for each point its premise value sits above the list's average (minus 3 for each point below). Promos with no premise rating get no premise adjustment. The heuristic acts as a validity filter and a fallback, and is not blended in. If some calls fail, the run continues and lists them in `llm_error`. If the provider rejects the key or model, the run falls back to the heuristic ranking.

Short rate limits are waited out automatically. If the provider asks for more than 90 seconds, which usually means a daily quota, the run stops calling it and is marked `hybrid_llm_partial`. Do not use a partial ranking, because scored promos rank ahead of unscored ones regardless of quality. Rerun the same command after the wait: finished replies come from the cache, so it continues where it stopped. Keep the same model for the whole run, since the cache is per model.

**Data notice:** LLM mode sends whole episode transcripts to the configured provider. Confirm this is allowed for confidential screenplays before enabling it. `.llm_cache` holds provider replies, which can quote the script, so keep it private.

## Tests

```bash
pytest -q
```

If `scripts.zip` is in the project root (or `PROMORANK_PACK` points to the pack), extra tests check the real output for lines outside their clip, nested copies and dirty cuts. They are skipped otherwise.

## Scoring (baseline)

| Dimension | Weight |
|---|---:|
| Opening hook | 30% |
| Ending cliffhanger / unresolved tension | 30% |
| Standalone clarity | 20% |
| Emotional intensity | 15% |
| Duration suitability | 5% |

Each dimension is scored from 0–5 and normalized to 100. These are initial engineering weights, not empirically learned weights. They should be calibrated through editor review.

## Output

`ranked_promos.csv` and `promorank_results.json` include:

- `segments`: every segment in order, as `[{episode, start, end}]`, including for single clips
- `hook`, `hook_at`, `hook_type`: the first beat on screen, when it happens, and its type
- `cliffhanger`: the last line fully inside the clip
- `why_it_works`, `quality_warning`, component scores, and `post_first_rank`

The JSON also has `post_first` (the launch list with reasons), `montages`, `quality_report`, and `run_metadata` (model, provider host, temperature, prompt version, settings and timestamp).

`quality_report.md` counts lines outside their clip, dirty cuts and overlaps, shows the shape of the ranking, lists where the heuristic and the LLM disagree most, and names the weakest endings in the list.

## Checking against hand-picked promos

Create a CSV outside the repo's tracked files, for example `data/gold_promos.csv`, with times in seconds or `MM:SS.ss`:

```csv
episode,start,end,note
1,11.0,41.5,The opening confrontation
```

Pass it with `--gold data/gold_promos.csv`. The report shows how many were picked and where the closest valid window ranks for the ones that were missed.

## Raw outputs

`.gitignore` keeps `.env`, `.venv`, `.llm_cache`, the screenplay pack and raw outputs out of this repository. Raw outputs are shared separately through a restricted link.

## Optional editor-calibrated ranker (after human review)

Export the ranked CSV, add an `editor_score` column (1–5 or 0–100) based on human review, and collect at least 20 genuinely reviewed candidates. Then install `requirements-training.txt` and run:

```bash
python -m promorank.train_ranker --labels raw_outputs/editor_review.csv --output models/editor_ranker.joblib
```

Do not fill `editor_score` with the system's own score; that would only teach the model to reproduce its existing heuristic.

## Known limitations

- Screenplay text is only a proxy for the finished video. Visual performance, pacing, acting, audio, and editability cannot be verified from text alone.
- Without the LLM, scoring relies on keywords and punctuation. It can overvalue dramatic words and can miss beats whose power comes from the story, such as a quiet proposal.
- Post-first assumes new viewers land on Episode 1. Spoiler risk there comes from story position alone, because the LLM's spoiler ratings did not vary across the series. Without an LLM, premise value is a rough estimate from the loglines.
- Montages are experimental. They take the closing beat of top promos from the start, middle and end of the series and may not read as one story.
- Character aliases are not merged, so one character credited under two names counts as two speakers.
- Timestamp precision and the relationship between screenplay timecodes and final rendered video should be checked before editing.
- The tool proposes promotional moments; it does not edit, add music, or create overlays.
