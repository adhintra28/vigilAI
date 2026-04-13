"""Vision + text analysis for VigilAI — grounded outputs and strict validation."""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from PIL import Image, ImageStat

Grounding = Literal["observed", "inferred", "speculative"]
RiskLevel = Literal["low", "medium", "high"]
Confidence = Literal["low", "medium", "high"]


class SuspiciousActivity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    confidence: Confidence = "medium"
    evidence: str = ""
    grounding: Grounding = "speculative"


class ObjectOfInterest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    role: str = ""
    notes: str = ""
    grounding: Grounding = "speculative"


class Anomaly(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str
    why_it_matters: str = ""
    grounding: Grounding = "speculative"


class Relationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    from_: str = Field(alias="from")
    relation: str = ""
    to: str = ""
    interpretation: str = ""
    grounding: Grounding = "speculative"


class OverlookedDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    detail: str
    why_easy_to_miss: str = ""
    grounding: Grounding = "speculative"


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    narrative: str
    supporting_cues: list[str] = Field(default_factory=list)
    alternatives: str = ""
    grounding: Grounding = "speculative"


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: RiskLevel = "low"
    rationale: str = ""
    grounding: Grounding = "speculative"


class AnalysisReport(BaseModel):
    """Strict shape after coercion — extra keys from the model are dropped."""

    model_config = ConfigDict(extra="ignore")

    scene_summary: str = ""
    visible_facts: list[str] = Field(default_factory=list)
    suspicious_activities: list[SuspiciousActivity] = Field(default_factory=list)
    objects_of_interest: list[ObjectOfInterest] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    overlooked_details: list[OverlookedDetail] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment)
    investigation_follow_up: list[str] = Field(default_factory=list)
    limitations_and_uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def limitations_not_empty(self) -> AnalysisReport:
        if not any(str(x).strip() for x in self.limitations_and_uncertainties):
            raise ValueError("limitations_and_uncertainties must contain at least one non-empty entry")
        return self


SYSTEM_PROMPT = """You are VigilAI, assisting security and investigations with image or narrative scene review.

Rules for accuracy and accountability:
1. visible_facts: ONLY statements directly supported by the image pixels or the explicit witness text. No guessing.
2. Every interpretive item (suspicious activities, objects_of_interest roles beyond naming visible objects, anomalies,
   relationships, overlooked_details, hypotheses, risk_assessment) MUST include grounding:
   - "observed" only if the claim is directly visible or explicitly stated in the narrative.
   - "inferred" for careful deductions tied to named visible_facts or quoted narrative.
   - "speculative" for possibilities; keep these clearly tentative.
3. If something is unclear (blur, occlusion, darkness), say so under limitations_and_uncertainties — do not invent detail.
4. Do not invent identities, exact times, license plates, or unread text you cannot read.
5. limitations_and_uncertainties must list at least one honest limitation (e.g. single frame, no audio, no depth).
6. Output a single JSON object only, no markdown fences."""


USER_SCHEMA = """Return JSON with exactly these keys:
{
  "scene_summary": "string — short; distinguish facts vs interpretation",
  "visible_facts": ["string — only direct observations from the image or stated in the narrative"],
  "suspicious_activities": [
    {"label": "string", "confidence": "low|medium|high", "evidence": "string", "grounding": "observed|inferred|speculative"}
  ],
  "objects_of_interest": [
    {"name": "string", "role": "string", "notes": "string", "grounding": "observed|inferred|speculative"}
  ],
  "anomalies": [
    {"description": "string", "why_it_matters": "string", "grounding": "observed|inferred|speculative"}
  ],
  "relationships": [
    {"from": "string", "relation": "string", "to": "string", "interpretation": "string", "grounding": "observed|inferred|speculative"}
  ],
  "overlooked_details": [
    {"detail": "string", "why_easy_to_miss": "string", "grounding": "observed|inferred|speculative"}
  ],
  "hypotheses": [
    {"narrative": "string", "supporting_cues": ["string"], "alternatives": "string", "grounding": "inferred|speculative"}
  ],
  "risk_assessment": {"level": "low|medium|high", "rationale": "string", "grounding": "observed|inferred|speculative"},
  "investigation_follow_up": ["string — checks that reduce uncertainty; avoid claiming certainty"],
  "limitations_and_uncertainties": ["string — at least one entry"]
}
Use empty arrays where nothing applies. Be concise."""


def _image_to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _to_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def prepare_image_for_groq(image_bytes: bytes, max_b64_len: int = 3_000_000) -> tuple[bytes, str]:
    """
    Re-encode as JPEG and shrink until base64 length is under Groq's safe limit
    (Groq: ~4MB cap on base64 image payloads — leave margin for JSON overhead).
    """
    im = _to_rgb(Image.open(BytesIO(image_bytes)))
    quality = 88
    scale = 1.0
    working = im
    last_raw: bytes = b""
    for _ in range(28):
        w, h = working.size
        buf = BytesIO()
        working.save(buf, format="JPEG", quality=int(quality), optimize=True)
        last_raw = buf.getvalue()
        b64_len = len(base64.standard_b64encode(last_raw))
        if b64_len <= max_b64_len:
            return last_raw, "image/jpeg"
        if quality > 52:
            quality -= 6
        else:
            scale *= 0.82
            nw = max(int(w * scale), 384)
            nh = max(int(h * scale), 384)
            working = working.resize((nw, nh), Image.Resampling.LANCZOS)
    return last_raw, "image/jpeg"


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def extract_factual_image_metadata(image_bytes: bytes) -> list[str]:
    """Deterministic, pixel-based facts only (no semantic labels)."""
    im = Image.open(BytesIO(image_bytes))
    facts = [
        f"Image dimensions: {im.width} × {im.height} pixels.",
        f"Color mode: {im.mode}.",
    ]
    if im.format:
        facts.append(f"Container/format hint from decoder: {im.format}.")
    gray = im.convert("L")
    stat = ImageStat.Stat(gray)
    mean_l = float(stat.mean[0])
    facts.append(f"Mean grayscale luminance (0–255): {mean_l:.1f}.")
    return facts


def _coerce_grounding(value: Any) -> Grounding:
    if value in ("observed", "inferred", "speculative"):
        return value  # type: ignore[return-value]
    return "speculative"


def _coerce_dict_for_report(data: dict[str, Any]) -> dict[str, Any]:
    """Fill safe defaults before Pydantic validation."""
    out = dict(data)
    out.setdefault("scene_summary", "")
    vf = out.get("visible_facts")
    if not isinstance(vf, list):
        vf = []
    out["visible_facts"] = [str(x) for x in vf]
    lu = out.get("limitations_and_uncertainties")
    if not isinstance(lu, list):
        lu = []
    out["limitations_and_uncertainties"] = [str(x) for x in lu]
    out.setdefault("suspicious_activities", [])
    out.setdefault("objects_of_interest", [])
    out.setdefault("anomalies", [])
    out.setdefault("relationships", [])
    out.setdefault("overlooked_details", [])
    out.setdefault("hypotheses", [])
    out.setdefault("investigation_follow_up", [])
    out.setdefault("limitations_and_uncertainties", [])

    ra = out.get("risk_assessment")
    if not isinstance(ra, dict):
        ra = {}
    ra.setdefault("level", "low")
    ra.setdefault("rationale", "")
    ra.setdefault("grounding", "speculative")
    ra["grounding"] = _coerce_grounding(ra.get("grounding"))
    out["risk_assessment"] = ra

    for item in out.get("suspicious_activities") or []:
        if isinstance(item, dict):
            item["grounding"] = _coerce_grounding(item.get("grounding"))
    for item in out.get("objects_of_interest") or []:
        if isinstance(item, dict):
            item["grounding"] = _coerce_grounding(item.get("grounding"))
    for item in out.get("anomalies") or []:
        if isinstance(item, dict):
            item["grounding"] = _coerce_grounding(item.get("grounding"))
    for item in out.get("relationships") or []:
        if isinstance(item, dict):
            item["grounding"] = _coerce_grounding(item.get("grounding"))
    for item in out.get("overlooked_details") or []:
        if isinstance(item, dict):
            item["grounding"] = _coerce_grounding(item.get("grounding"))
    for item in out.get("hypotheses") or []:
        if isinstance(item, dict):
            g = _coerce_grounding(item.get("grounding"))
            if g == "observed":
                g = "inferred"
            item["grounding"] = g

    return out


def validate_report(data: dict[str, Any]) -> AnalysisReport:
    return AnalysisReport.model_validate(_coerce_dict_for_report(data))


def report_to_display_dict(report: AnalysisReport) -> dict[str, Any]:
    d = report.model_dump(by_alias=True)
    return d


def _chat_json(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.15,
        max_completion_tokens=4096,
    )
    return resp.choices[0].message.content or "{}"


def _run_vision_analysis(
    client: OpenAI,
    model: str,
    user_content: list[dict[str, Any]],
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw = _chat_json(client, model, messages)
    try:
        parsed = _parse_json_object(raw)
        report = validate_report(parsed)
        return report_to_display_dict(report)
    except (json.JSONDecodeError, ValidationError) as first_err:
        repair = (
            "Your previous reply was not valid JSON or did not match the required schema. "
            f"Error (truncated): {str(first_err)[:800]}\n"
            "Reply again with ONE JSON object only, all required top-level keys, "
            "limitations_and_uncertainties non-empty, and every interpretive list item "
            "including grounding."
        )
        messages.append({"role": "assistant", "content": raw[:12000]})
        messages.append({"role": "user", "content": repair})
        raw2 = _chat_json(client, model, messages)
        try:
            parsed2 = _parse_json_object(raw2)
            report2 = validate_report(parsed2)
            return report_to_display_dict(report2)
        except (json.JSONDecodeError, ValidationError) as second_err:
            raise ValueError(
                "Model returned invalid JSON twice after repair. "
                f"First error: {first_err!s}; second: {second_err!s}"
            ) from second_err


def analyze_image(
    client: OpenAI,
    image_bytes: bytes,
    image_format: str | None,
    extra_context: str | None,
    model: str,
) -> dict[str, Any]:
    _ = image_format  # original format; we re-encode as JPEG for Groq payload limits
    jpeg_bytes, mime = prepare_image_for_groq(image_bytes)
    url = _image_to_data_url(jpeg_bytes, mime)
    user_parts: list[dict[str, Any]] = [
        {"type": "text", "text": USER_SCHEMA},
        {"type": "image_url", "image_url": {"url": url}},
    ]
    if extra_context:
        user_parts.insert(
            0,
            {
                "type": "text",
                "text": (
                    "Investigator context (may contain unverified claims; do not treat as visual evidence unless "
                    "consistent with pixels): "
                    + extra_context
                ),
            },
        )
    return _run_vision_analysis(client, model, user_parts)


def analyze_simulated_scene(
    client: OpenAI,
    scene_description: str,
    model: str,
) -> dict[str, Any]:
    text = f"""SIMULATED or WITNESS narrative scene (no image). visible_facts must only restate what is explicitly
written below — not invented sensory detail.

Narrative:
---
{scene_description}
---

{USER_SCHEMA}"""
    return _run_vision_analysis(
        client,
        model,
        [{"type": "text", "text": text}],
    )


def accurate_offline_image_report(image_bytes: bytes) -> dict[str, Any]:
    """No API: only deterministic metadata — no semantic crime labels."""
    facts = extract_factual_image_metadata(image_bytes)
    return {
        "scene_summary": (
            "Offline accurate mode: only machine-readable image properties are shown below. "
            "Semantic interpretation requires an API key and model."
        ),
        "visible_facts": facts,
        "suspicious_activities": [],
        "objects_of_interest": [],
        "anomalies": [],
        "relationships": [],
        "overlooked_details": [],
        "hypotheses": [],
        "risk_assessment": {
            "level": "low",
            "rationale": "No interpretive model run.",
            "grounding": "observed",
        },
        "investigation_follow_up": [
            "Configure GROQ_API_KEY (or Streamlit secrets) to run grounded multimodal analysis.",
        ],
        "limitations_and_uncertainties": [
            "Single static frame: no motion, audio, or multi-camera correlation.",
            "No vision-language analysis was executed in offline mode.",
        ],
    }


def accurate_offline_narrative_stub() -> dict[str, Any]:
    return {
        "scene_summary": (
            "Offline accurate mode: narrative text was not analyzed because no API key is configured."
        ),
        "visible_facts": [],
        "suspicious_activities": [],
        "objects_of_interest": [],
        "anomalies": [],
        "relationships": [],
        "overlooked_details": [],
        "hypotheses": [],
        "risk_assessment": {
            "level": "low",
            "rationale": "No language model run.",
            "grounding": "observed",
        },
        "investigation_follow_up": [
            "Add GROQ_API_KEY to obtain structured narrative analysis with explicit grounding fields.",
        ],
        "limitations_and_uncertainties": [
            "Without a model call, no inferences can be drawn from the supplied text.",
        ],
    }


def illustrative_demo_payload(mode: str) -> dict[str, Any]:
    """Clearly fictional sample for pitches — not derived from user data."""
    if mode == "image":
        return {
            "scene_summary": (
                "ILLUSTRATIVE SAMPLE ONLY — not generated from your upload. "
                "Generic street scene with a figure near a vehicle and items on the ground."
            ),
            "visible_facts": [
                "ILLUSTRATIVE: sample visible_facts line (not from your file).",
            ],
            "suspicious_activities": [
                {
                    "label": "Loitering near vehicle (sample)",
                    "confidence": "medium",
                    "evidence": "Sample row for UI demo.",
                    "grounding": "speculative",
                }
            ],
            "objects_of_interest": [
                {
                    "name": "Parked vehicle (sample)",
                    "role": "Sample",
                    "notes": "Replace with real analysis.",
                    "grounding": "speculative",
                }
            ],
            "anomalies": [
                {
                    "description": "Sample anomaly row",
                    "why_it_matters": "Demo only.",
                    "grounding": "speculative",
                }
            ],
            "relationships": [
                {
                    "from": "Person (sample)",
                    "relation": "proximate_to",
                    "to": "Objects on ground (sample)",
                    "interpretation": "Demo relationship.",
                    "grounding": "speculative",
                }
            ],
            "overlooked_details": [
                {
                    "detail": "Shadows / reflections (sample prompt)",
                    "why_easy_to_miss": "Demo.",
                    "grounding": "speculative",
                }
            ],
            "hypotheses": [
                {
                    "narrative": "Sample hypothesis narrative.",
                    "supporting_cues": ["Sample"],
                    "alternatives": "Demo alternatives.",
                    "grounding": "speculative",
                }
            ],
            "risk_assessment": {
                "level": "medium",
                "rationale": "Illustrative sample — not an assessment of your data.",
                "grounding": "speculative",
            },
            "investigation_follow_up": [
                "Replace with real API-backed follow-ups after configuring GROQ_API_KEY.",
            ],
            "limitations_and_uncertainties": [
                "This entire block is a static UI sample, not model output.",
            ],
        }
    return {
        "scene_summary": (
            "ILLUSTRATIVE SAMPLE ONLY — not generated from your narrative. "
            "Retail back-corridor after-hours scenario."
        ),
        "visible_facts": ["ILLUSTRATIVE: not from your text."],
        "suspicious_activities": [
            {
                "label": "After-hours access (sample)",
                "confidence": "high",
                "evidence": "Demo row.",
                "grounding": "speculative",
            }
        ],
        "objects_of_interest": [
            {
                "name": "Propped door (sample)",
                "role": "Sample",
                "notes": "Demo.",
                "grounding": "speculative",
            }
        ],
        "anomalies": [],
        "relationships": [
            {
                "from": "Door (sample)",
                "relation": "enables",
                "to": "Movement (sample)",
                "interpretation": "Demo.",
                "grounding": "speculative",
            }
        ],
        "overlooked_details": [
            {
                "detail": "Egress signage (sample)",
                "why_easy_to_miss": "Demo.",
                "grounding": "speculative",
            }
        ],
        "hypotheses": [
            {
                "narrative": "Insider-assisted entry (sample).",
                "supporting_cues": ["Demo"],
                "alternatives": "Alarm egress (sample).",
                "grounding": "speculative",
            }
        ],
        "risk_assessment": {
            "level": "medium",
            "rationale": "Illustrative sample only.",
            "grounding": "speculative",
        },
        "investigation_follow_up": ["Correlate logs (sample)."],
        "limitations_and_uncertainties": [
            "Static sample — not produced from your narrative.",
        ],
    }
