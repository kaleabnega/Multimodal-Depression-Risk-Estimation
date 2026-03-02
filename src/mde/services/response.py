from __future__ import annotations

from mde.core.types import AgentInput, PolicyState


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
