from __future__ import annotations

import re
from typing import Optional

from mde.core.types import AgentInput, PolicyState

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional runtime dependency
    InferenceClient = None


class TemplateResponseGenerator:
    """Policy-gated response templates with optional cue summaries."""

    def __init__(self) -> None:
        self.crisis_line = (
            "I am really glad you shared this. You may be in immediate danger, and it is important to contact "
            "local emergency services or a crisis hotline right now. If you are in the U.S., call or text 988."
        )

    def generate(self, data: AgentInput) -> str:
        if data.policy_state == PolicyState.CRISIS_PROTOCOL:
            return self.crisis_line

        cue_parts: list[str] = []
        if data.audio_summary and data.audio_summary != "no audio cues":
            cue_parts.append(f"audio: {data.audio_summary}")
        if data.visual_summary and data.visual_summary != "no visual cues":
            cue_parts.append(f"visual: {data.visual_summary}")

        cue_text = ""
        if cue_parts:
            cue_text = " I also noticed " + "; ".join(cue_parts) + "."

        if data.policy_state == PolicyState.HIGH_RISK_SUPPORT:
            return (
                "Thank you for sharing this. It sounds like you are carrying a lot right now. "
                "You deserve support from a licensed mental health professional, and reaching out today could help." + cue_text
            )

        if data.policy_state == PolicyState.GENTLE_MONITORING:
            return (
                "I hear you. If you want, we can slow this down and talk through what has felt heaviest lately. "
                "I can also share coping steps and support options." + cue_text
            )

        return (
            "Thanks for opening up. I can help with reflective prompts, coping strategies, "
            "or practical mental health resources based on what you need next."
        )


class GuardedLLMResponseGenerator:
    """LLM responder with policy gating and hard safety validation."""

    def __init__(
        self,
        model_name: str = "HuggingFaceH4/zephyr-7b-beta",
        api_token: Optional[str] = None,
        allow_fallback: bool = True,
        use_llm_for_high_risk: bool = False,
    ) -> None:
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self.use_llm_for_high_risk = use_llm_for_high_risk
        self.template = TemplateResponseGenerator()
        self.client = InferenceClient(token=api_token) if InferenceClient is not None else None

        self._unsafe_patterns = [
            r"\bdiagnos(?:e|ed|is)\b",
            r"\byou have depression\b",
            r"\bkill yourself\b",
            r"\bend your life\b",
            r"\bself[- ]?harm (?:method|steps|instructions)\b",
            r"\boverdose\b",
            r"\bsuicide plan\b",
        ]

    def _prompt(self, data: AgentInput) -> str:
        cue_parts: list[str] = []
        if data.audio_summary and data.audio_summary != "no audio cues":
            cue_parts.append(f"audio={data.audio_summary}")
        if data.visual_summary and data.visual_summary != "no visual cues":
            cue_parts.append(f"visual={data.visual_summary}")
        cues = ", ".join(cue_parts) if cue_parts else "none"

        return (
            "You are a mental-health support assistant.\n"
            "Rules:\n"
            "1) Be empathetic and brief (3-5 sentences).\n"
            "2) Do not diagnose.\n"
            "3) Do not provide self-harm instructions.\n"
            "4) Suggest professional support when risk is high.\n"
            "5) Use plain language.\n\n"
            f"Policy state: {data.policy_state.value}\n"
            f"Risk score: {data.risk_score:.3f}\n"
            f"Observed cues: {cues}\n"
            f"User message: {data.user_text}\n\n"
            "Assistant response:"
        )

    def _is_safe(self, text: str, state: PolicyState) -> bool:
        if not text.strip():
            return False
        if len(text) > 1200:
            return False

        lowered = text.lower()
        for pattern in self._unsafe_patterns:
            if re.search(pattern, lowered):
                return False

        if state == PolicyState.HIGH_RISK_SUPPORT:
            mentions_support = any(
                token in lowered for token in ["professional", "therapist", "counselor", "support line", "988"]
            )
            if not mentions_support:
                return False

        return True

    def _generate_llm(self, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError("huggingface_hub is not installed")

        output = self.client.text_generation(
            prompt,
            model=self.model_name,
            max_new_tokens=220,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
        )
        text = output if isinstance(output, str) else str(output)
        if "Assistant response:" in text:
            text = text.split("Assistant response:", 1)[1]
        return text.strip()

    def generate(self, data: AgentInput) -> str:
        # Crisis path is deterministic and never delegated to LLM.
        if data.policy_state == PolicyState.CRISIS_PROTOCOL:
            return self.template.generate(data)

        if data.policy_state == PolicyState.HIGH_RISK_SUPPORT and not self.use_llm_for_high_risk:
            return self.template.generate(data)

        try:
            generated = self._generate_llm(self._prompt(data))
            if self._is_safe(generated, data.policy_state):
                return generated
            if not self.allow_fallback:
                raise ValueError("LLM output did not pass safety validation")
            return self.template.generate(data)
        except Exception:
            if not self.allow_fallback:
                raise
            return self.template.generate(data)
