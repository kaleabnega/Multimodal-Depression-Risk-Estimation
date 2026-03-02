from __future__ import annotations

import re

from mde.core.types import PolicyDecision, PolicyState


class SafetyPolicyEngine:
    """Threshold + crisis-keyword policy decision layer."""

    def __init__(self, low: float = 0.30, high: float = 0.70) -> None:
        self.low = low
        self.high = high
        self.crisis_patterns = [
            r"\bkill myself\b",
            r"\bend my life\b",
            r"\bsuicid\w*\b",
            r"\bwant to die\b",
            r"\bhurt myself\b",
        ]

    def _has_crisis_language(self, text: str) -> bool:
        lowered = text.lower()
        return any(re.search(pat, lowered) for pat in self.crisis_patterns)

    def decide(self, text: str, risk_score: float) -> PolicyDecision:
        reasons: list[str] = []

        if self._has_crisis_language(text):
            reasons.append("explicit crisis language detected")
            return PolicyDecision(risk_score=risk_score, state=PolicyState.CRISIS_PROTOCOL, reasons=reasons)

        if risk_score >= self.high:
            reasons.append(f"risk score >= high threshold ({self.high:.2f})")
            state = PolicyState.HIGH_RISK_SUPPORT
        elif risk_score >= self.low:
            reasons.append(f"risk score between thresholds ({self.low:.2f}, {self.high:.2f})")
            state = PolicyState.GENTLE_MONITORING
        else:
            reasons.append(f"risk score < low threshold ({self.low:.2f})")
            state = PolicyState.NORMAL_SUPPORT

        return PolicyDecision(risk_score=risk_score, state=state, reasons=reasons)
