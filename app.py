"""
VigilAI — Visual crime analysis assistant.
Run: streamlit run app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st
from pydantic import ValidationError
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from PIL import Image

from analysis import (
    accurate_offline_image_report,
    accurate_offline_narrative_stub,
    analyze_image,
    analyze_simulated_scene,
    illustrative_demo_payload,
)

load_dotenv()

st.set_page_config(
    page_title="VigilAI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.getenv(
    "VIGILAI_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)


def get_client() -> OpenAI | None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except (FileNotFoundError, KeyError, RuntimeError, TypeError):
            key = None
    if not key:
        return None
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL)


def _g(grounding: str | None) -> str:
    g = (grounding or "speculative").lower()
    if g not in ("observed", "inferred", "speculative"):
        g = "speculative"
    return g


def _grounding_caption(grounding: str | None) -> str:
    return f"Grounding: **{_g(grounding)}** — observed = directly seen/stated; inferred = tied to facts; speculative = tentative."


def render_result(data: dict) -> None:
    st.subheader("Scene summary")
    st.write(data.get("scene_summary", "—"))

    facts = data.get("visible_facts") or []
    if facts:
        st.markdown("#### Visible facts (evidence-first)")
        for line in facts:
            st.markdown(f"- {line}")

    lim = data.get("limitations_and_uncertainties") or []
    if lim:
        with st.expander("Limitations and uncertainties", expanded=True):
            for line in lim:
                st.markdown(f"- {line}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Suspicious activities")
        for item in data.get("suspicious_activities") or []:
            st.markdown(f"**{item.get('label', '')}** `{item.get('confidence', '')}`")
            st.caption(item.get("evidence", ""))
            st.caption(_grounding_caption(item.get("grounding")))
        if not data.get("suspicious_activities"):
            st.caption("None flagged.")

        st.markdown("#### Objects of interest")
        for o in data.get("objects_of_interest") or []:
            st.markdown(f"**{o.get('name', '')}** — _{o.get('role', '')}_")
            st.caption(o.get("notes", ""))
            st.caption(_grounding_caption(o.get("grounding")))
        if not data.get("objects_of_interest"):
            st.caption("None listed.")

    with c2:
        st.markdown("#### Anomalies")
        for a in data.get("anomalies") or []:
            st.markdown(f"**{a.get('description', '')}**")
            st.caption(a.get("why_it_matters", ""))
            st.caption(_grounding_caption(a.get("grounding")))
        if not data.get("anomalies"):
            st.caption("None flagged.")

        st.markdown("#### Relationships")
        for r in data.get("relationships") or []:
            st.markdown(
                f"**{r.get('from', '')}** _{r.get('relation', '')}_ **{r.get('to', '')}**"
            )
            st.caption(r.get("interpretation", ""))
            st.caption(_grounding_caption(r.get("grounding")))
        if not data.get("relationships"):
            st.caption("None listed.")

    st.markdown("#### Easily overlooked details")
    for d in data.get("overlooked_details") or []:
        st.markdown(f"**{d.get('detail', '')}**")
        st.caption(d.get("why_easy_to_miss", ""))
        st.caption(_grounding_caption(d.get("grounding")))
    if not data.get("overlooked_details"):
        st.caption("None listed.")

    st.markdown("#### Hypotheses (what might have happened)")
    for h in data.get("hypotheses") or []:
        st.markdown(f"_{h.get('narrative', '')}_")
        cues = h.get("supporting_cues") or []
        if cues:
            st.caption("Cues: " + ", ".join(str(c) for c in cues))
        st.caption("Alternatives: " + str(h.get("alternatives", "")))
        st.caption(_grounding_caption(h.get("grounding")))
    if not data.get("hypotheses"):
        st.caption("None listed.")

    risk = data.get("risk_assessment") or {}
    level = str(risk.get("level", "—")).upper()
    st.markdown(f"#### Risk assessment: **{level}**")
    st.write(risk.get("rationale", ""))
    st.caption(_grounding_caption(risk.get("grounding")))

    st.markdown("#### Investigation / monitoring follow-up")
    for step in data.get("investigation_follow_up") or []:
        st.markdown(f"- {step}")
    if not data.get("investigation_follow_up"):
        st.caption("None listed.")


def main() -> None:
    st.title("VigilAI")
    st.caption(
        "Evidence-first scene review: visible facts, explicit grounding, and honest limits — "
        "not a substitute for lab procedures or legal findings."
    )

    with st.sidebar:
        st.header("Configuration")
        model = st.text_input(
            "Groq model id",
            value=DEFAULT_MODEL,
            help="Vision + JSON on Groq, e.g. meta-llama/llama-4-scout-17b-16e-instruct",
        )
        use_demo = st.checkbox(
            "Force offline mode (no API calls)",
            value=False,
            help="Skips the model entirely.",
        )
        illustrative = st.checkbox(
            "Illustrative UI sample (not from your data)",
            value=False,
            help="When offline, show a clearly labeled static example instead of metadata-only accurate mode.",
        )
        st.divider()

    client = None if use_demo else get_client()
    if not use_demo and client is None:
        st.warning(
            "No `GROQ_API_KEY` found — running in **offline** mode (deterministic image metadata only, "
            "or illustrative sample if you enable it)."
        )

    tab_img, tab_sim = st.tabs(["Image / scene photo", "Simulated scene (text)"])

    with tab_img:
        uploaded = st.file_uploader(
            "Upload an image (CCTV still, phone photo, forensic frame)",
            type=["png", "jpg", "jpeg", "webp", "gif"],
        )
        ctx = st.text_area(
            "Optional investigator context (time, location hint, known events)",
            height=80,
            key="img_ctx",
        )
        run_img = st.button("Analyze image", type="primary", disabled=uploaded is None)

        if uploaded is not None:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)

        if run_img and uploaded is not None:
            buf = uploaded.getvalue()
            fmt = img.format
            with st.spinner("Analyzing…"):
                if client is None:
                    if illustrative:
                        st.warning("Showing **illustrative sample** rows — they are not inferred from your file.")
                        out = illustrative_demo_payload("image")
                    else:
                        st.info(
                            "**Accurate offline mode:** only pixel-level metadata is shown; "
                            "no semantic labels are guessed without a model."
                        )
                        out = accurate_offline_image_report(buf)
                else:
                    try:
                        out = analyze_image(client, buf, fmt, ctx or None, model)
                    except (OpenAIError, ValueError, json.JSONDecodeError, ValidationError) as e:
                        st.error(f"Analysis failed: {e}")
                        out = None
            if out is not None:
                render_result(out)

    with tab_sim:
        scene = st.text_area(
            "Describe the simulated or witness-reported scene in detail",
            height=200,
            placeholder="Example: Loading dock, 02:40, roll-up door cracked 30cm, interior lights off, "
            "forklift moved 3m from usual bay, security seal tape on crate cut…",
        )
        run_sim = st.button("Analyze simulated scene", type="primary", disabled=not (scene or "").strip())

        if run_sim and scene.strip():
            with st.spinner("Analyzing…"):
                if client is None:
                    if illustrative:
                        st.warning("Showing **illustrative sample** — not produced from your narrative.")
                        out = illustrative_demo_payload("simulated")
                    else:
                        st.info(
                            "**Accurate offline mode:** narrative was not interpreted without a model."
                        )
                        out = accurate_offline_narrative_stub()
                else:
                    try:
                        out = analyze_simulated_scene(client, scene.strip(), model)
                    except (OpenAIError, ValueError, json.JSONDecodeError, ValidationError) as e:
                        st.error(f"Analysis failed: {e}")
                        out = None
            if out is not None:
                render_result(out)

    st.divider()
    st.caption(
        "Multimodal models can still err: treat outputs as triage aids and verify against primary evidence."
    )


if __name__ == "__main__":
    main()
