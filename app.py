from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from promorank.parser import load_zip_bytes
from promorank.pipeline import results_to_csv, run_pipeline
from promorank.quality import report_to_markdown

st.set_page_config(page_title="PromoRank AI", page_icon="🎬", layout="wide")
st.title("🎬 PromoRank AI")
st.caption("Discover and rank 20-60 second microdrama promos directly from timestamped screenplays.")

with st.sidebar:
    st.header("Analysis settings")
    max_results = st.slider("Promos to return", min_value=10, max_value=100, value=100, step=5)
    max_per_episode = st.slider("Maximum promos per episode", min_value=1, max_value=5, value=3)
    use_llm = st.checkbox(
        "Use optional LLM narrative evaluation", value=False,
        help="Reads OPENAI_* or GROQ_* settings from the .env file. "
             "Whole episode transcripts are sent to that provider.",
    )
    llm_options = {"cache_dir": ".llm_cache"}
    if use_llm:
        st.caption("Whole episode transcripts go to the provider to find beats, then clips are scored and the top 30 compared.")
    else:
        st.caption("Without an API key, the deterministic scoring baseline works locally.")

uploaded = st.file_uploader("Upload the screenplay ZIP", type=["zip"])
if uploaded:
    st.info(
        "The ZIP is processed in memory and is not saved by this app. "
        + ("Episode transcripts will be sent to the configured LLM provider, and replies are cached in .llm_cache." if use_llm else "Nothing leaves this machine.")
    )
    if st.button("Analyze all episodes", type="primary", width="stretch"):
        try:
            bar = st.progress(0.0, text="Parsing screenplays and scoring candidate windows...") if use_llm else None

            def progress(done: int, total: int) -> None:
                if bar is not None:
                    bar.progress(done / total, text=f"LLM step: {done}/{total}")

            with st.spinner("Parsing screenplays and evaluating candidate windows..."):
                episodes = load_zip_bytes(uploaded.getvalue())
                results = run_pipeline(
                    episodes,
                    max_results=max_results,
                    max_per_episode=max_per_episode,
                    use_llm=use_llm,
                    progress=progress,
                    llm_options=llm_options,
                )
            if bar is not None:
                bar.empty()
            st.session_state["promorank_results"] = results
            st.success(f"Analyzed {results['dataset']['episode_count']} episodes and generated {results['candidate_count']:,} candidate windows.")
        except Exception as exc:
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

results = st.session_state.get("promorank_results")
if results:
    dataset = results["dataset"]
    a, b, c, d = st.columns(4)
    a.metric("Episodes", dataset["episode_count"])
    b.metric("Scenes", dataset["scene_count"])
    c.metric("Timestamped blocks", dataset["event_count"])
    d.metric("Candidate windows", f"{results['candidate_count']:,}")
    mode = results["model_mode"]
    if results.get("llm_used"):
        mode += f" · {results.get('llm_evaluated_count', 0)} candidates evaluated by the LLM"
    st.caption(f"Scoring mode: {mode}")
    if results.get("llm_error"):
        st.warning(results["llm_error"])
    for warning in dataset.get("parse_warnings", []):
        st.warning(f"Parsing: {warning}")

    promos = results["ranked_promos"]
    if promos:
        if results.get("post_first"):
            st.subheader("Post these first")
            st.caption("Assumes new viewers land on Episode 1, so early, low-spoiler promos are favoured.")
            st.dataframe(pd.DataFrame([{
                "Order": t["post_first_rank"],
                "Overall rank": t["rank"],
                "Episode": t["episode"],
                "Time range": t["time_range"],
                "Hook type": t["hook_type"],
                "Why first": t["reason"],
            } for t in results["post_first"]]), width="stretch", hide_index=True)

        df = pd.DataFrame([{
            "Rank": p["rank"],
            "Episode": p["episode"],
            "Time range": p["time_range"],
            "Duration (s)": p["duration"],
            "Score": p["final_score"],
            "Source": p.get("evaluation_source", "heuristic"),
            "Hook": f"{p['hook_at']:.1f}s: {p['hook'][:100]}",
            "Cliffhanger": p["cliffhanger"][:100],
        } for p in promos])
        st.subheader("Ranked promo recommendations")
        st.dataframe(df, width="stretch", hide_index=True)
        for promo in promos:
            with st.expander(f"#{promo['rank']} · Episode {promo['episode']} · {promo['time_range']} · {promo['final_score']:.1f}/100"):
                st.markdown(f"**Opening hook** ({promo['hook_type']}, at {promo['hook_at']:.1f}s)\n\n{promo['hook'] or 'No clear opening hook detected.'}")
                st.markdown(f"**Ending / cliffhanger**\n\n{promo['cliffhanger'] or 'No clear ending beat detected.'}")
                st.markdown(f"**Why it may work**\n\n{promo['why_it_works']}")
                st.markdown(f"**Quality warning**\n\n{promo['quality_warning']}")
                st.markdown("**Dialogue excerpt**")
                st.text(promo["dialogue_excerpt"])
                st.caption(
                    f"Heuristic scores · Hook {promo['hook_score']}/5 · Cliffhanger {promo['cliffhanger_score']}/5 · "
                    f"Standalone {promo['standalone_score']}/5 · Emotion {promo['emotion_score']}/5 · "
                    f"Total {promo['heuristic_score']:.1f}/100"
                )
                if promo.get("llm_score") is not None:
                    st.caption(
                        f"LLM scores · Hook {promo['llm_hook']:.0f}/5 · Cliffhanger {promo['llm_cliffhanger']:.0f}/5 · "
                        f"Standalone {promo['llm_standalone']:.0f}/5 · Emotion {promo['llm_emotion']:.0f}/5 · "
                        f"Total {promo['llm_score']:.1f}/100"
                    )

        if results.get("montages"):
            st.subheader("Cross-episode montages (experimental)")
            for montage in results["montages"]:
                with st.expander(f"{montage['montage_id']} · {montage['total_duration']:.1f}s · {montage['score']:.1f}/100"):
                    st.markdown(montage["why_it_works"])
                    for segment in montage["segments"]:
                        st.markdown(f"- **Ep {segment['episode']}** {segment['time_range']}: {segment['beat']}")
                    st.caption(montage["quality_warning"])

        with st.expander("Quality report: where this run looks weak"):
            st.markdown(report_to_markdown(results))

        st.download_button(
            "Download ranked promos (CSV)",
            data=results_to_csv(results).encode("utf-8-sig"),
            file_name="ranked_promos.csv",
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            "Download complete results (JSON)",
            data=json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="promorank_results.json",
            mime="application/json",
            width="stretch",
        )
    else:
        st.warning("No candidates passed the minimum dialogue, duration and clean-cut filters. Check timestamp parsing and input format.")
else:
    st.markdown("### What this prototype does")
    st.markdown(
        "1. Parses episode, scene, dialogue, action, camera and sound-effect timestamps.\n"
        "2. Builds 20–60 second windows on shot boundaries that never cut through a line or an action.\n"
        "3. Scores only what the viewer fully sees: hook, ending tension, standalone clarity, emotion and duration.\n"
        "4. Rejects overlapping clips, suggests what to post first, and audits its own output."
    )
    st.warning("This is a first-pass decision-support tool, not an editor replacement. Review every timestamp and candidate against the finished video before publishing.")
