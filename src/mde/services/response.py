from __future__ import annotations

import re
from typing import Any, Optional

from mde.core.types import AgentInput, PolicyState

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional runtime dependency
    InferenceClient = None


def _is_visual_expression_query(text: str) -> bool:
    lowered = text.lower()
    has_visual_reference = any(token in lowered for token in ["face", "facial", "expression", "look", "smile", "video"])
    has_question_intent = any(token in lowered for token in ["how", "what", "do i", "am i", "?"])
    return has_visual_reference and has_question_intent


def _format_visual_expression_response(affect_probs: Optional[list[float]]) -> Optional[str]:
    if not affect_probs or len(affect_probs) < 3:
        return "I do not have enough visual signal to estimate your facial expression reliably from this input."

    sad, neutral, engaged = [max(0.0, float(x)) for x in affect_probs[:3]]
    total = sad + neutral + engaged
    if total <= 0.0:
        return "I do not have enough visual signal to estimate your facial expression reliably from this input."

    sad /= total
    neutral /= total
    engaged /= total
    probs = [sad, neutral, engaged]
    labels = ["sad/low-affect", "neutral", "positive/engaged"]
    best_idx = max(range(3), key=lambda i: probs[i])
    confidence = probs[best_idx]

    if confidence < 0.50:
        return (
            "The visual signal is mixed, so confidence is low. "
            f"Estimated distribution: sad/low-affect {sad:.0%}, neutral {neutral:.0%}, positive/engaged {engaged:.0%}."
        )

    return (
        f"From the visual cues, your expression appears mostly {labels[best_idx]} "
        f"(confidence {confidence:.0%}). "
        f"Estimated distribution: sad/low-affect {sad:.0%}, neutral {neutral:.0%}, positive/engaged {engaged:.0%}."
    )


def _visual_affect_context(affect_probs: Optional[list[float]]) -> dict[str, Any]:
    if not affect_probs or len(affect_probs) < 3:
        return {
            "available": False,
            "dominant_label": "unknown",
            "confidence": 0.0,
            "distribution": {"sad_low_affect": 0.0, "neutral": 0.0, "positive_engaged": 0.0},
        }

    sad, neutral, engaged = [max(0.0, float(x)) for x in affect_probs[:3]]
    total = sad + neutral + engaged
    if total <= 0.0:
        return {
            "available": False,
            "dominant_label": "unknown",
            "confidence": 0.0,
            "distribution": {"sad_low_affect": 0.0, "neutral": 0.0, "positive_engaged": 0.0},
        }

    sad /= total
    neutral /= total
    engaged /= total
    probs = [sad, neutral, engaged]
    labels = ["sad/low-affect", "neutral", "positive/engaged"]
    best_idx = max(range(3), key=lambda i: probs[i])
    return {
        "available": True,
        "dominant_label": labels[best_idx],
        "confidence": probs[best_idx],
        "distribution": {"sad_low_affect": sad, "neutral": neutral, "positive_engaged": engaged},
    }


class TemplateResponseGenerator:
    """Policy-gated response templates with optional cue summaries."""

    def __init__(self) -> None:
        self.crisis_line = (
            "I am really glad you shared this. You may be in immediate danger, and it is important to contact "
            "local emergency services or a crisis hotline right now. If you are in the U.S., call or text 988."
        )

    def generate(self, data: AgentInput) -> str:
        if _is_visual_expression_query(data.user_text):
            visual_answer = _format_visual_expression_response(data.visual_affect_probs)
            if visual_answer:
                return visual_answer

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
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        api_token: Optional[str] = None,
        allow_fallback: bool = True,
        use_llm_for_high_risk: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_candidates = [
            model_name,
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ]
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
        visual_ctx = _visual_affect_context(data.visual_affect_probs)
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
            "If user asks specifically about facial expression, answer from visual_affect context "
            "with confidence and avoid depression conclusions.\n\n"
            f"Policy state: {data.policy_state.value}\n"
            f"Risk score: {data.risk_score:.3f}\n"
            f"Observed cues: {cues}\n"
            f"Visual affect context: {visual_ctx}\n"
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

        # Prefer conversational endpoints because some providers expose chat-only tasks
        # for instruct/chat models (e.g., Zephyr).
        chat_messages = [
            {
                "role": "system",
                "content": (
                    "You are a safe mental-health support assistant. "
                    "Do not diagnose and do not provide self-harm instructions."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        last_error: Exception | None = None
        for model_name in self.model_candidates:
            try:
                if hasattr(self.client, "chat_completion"):
                    response = self.client.chat_completion(
                        model=model_name,
                        messages=chat_messages,
                        max_tokens=220,
                        temperature=0.4,
                        top_p=0.9,
                    )
                    text = self._extract_chat_text(response)
                    if text:
                        return text.strip()
            except Exception as exc:  # pragma: no cover - depends on runtime client/provider
                last_error = exc

            try:
                chat_api = getattr(self.client, "chat", None)
                completions = getattr(chat_api, "completions", None) if chat_api else None
                create = getattr(completions, "create", None) if completions else None
                if callable(create):
                    response = create(
                        model=model_name,
                        messages=chat_messages,
                        max_tokens=220,
                        temperature=0.4,
                        top_p=0.9,
                    )
                    text = self._extract_chat_text(response)
                    if text:
                        return text.strip()
            except Exception as exc:  # pragma: no cover - depends on runtime client/provider
                last_error = exc

            try:
                output = self.client.text_generation(
                    prompt,
                    model=model_name,
                    max_new_tokens=220,
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    do_sample=True,
                )
                text = output if isinstance(output, str) else str(output)
                if "Assistant response:" in text:
                    text = text.split("Assistant response:", 1)[1]
                text = text.strip()
                if text:
                    return text
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                "No supported response model found for your enabled HF providers. "
                "Set a supported model with --response-model."
            ) from last_error
        raise RuntimeError("Unable to generate LLM response")

    def _extract_chat_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                return "".join(parts).strip()

        if isinstance(response, dict):
            raw_choices = response.get("choices", [])
            if raw_choices:
                msg = raw_choices[0].get("message", {})
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                    return "".join(parts).strip()

        return ""

    def generate(self, data: AgentInput) -> str:
        if _is_visual_expression_query(data.user_text):
            visual_ctx = _visual_affect_context(data.visual_affect_probs)
            if not visual_ctx["available"] or visual_ctx["confidence"] < 0.50:
                visual_answer = _format_visual_expression_response(data.visual_affect_probs)
                if visual_answer:
                    return visual_answer

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
