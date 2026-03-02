from mde.core.types import AgentInput, PolicyState
from mde.services.response import GuardedLLMResponseGenerator


def _agent_input(state: PolicyState) -> AgentInput:
    return AgentInput(
        user_text="I feel very low today",
        risk_score=0.55,
        policy_state=state,
        audio_summary=None,
        visual_summary=None,
        visual_affect_probs=None,
    )


def test_crisis_path_bypasses_llm() -> None:
    responder = GuardedLLMResponseGenerator(allow_fallback=True)
    out = responder.generate(_agent_input(PolicyState.CRISIS_PROTOCOL))
    assert "988" in out


def test_unsafe_llm_output_falls_back_to_template() -> None:
    responder = GuardedLLMResponseGenerator(allow_fallback=True)
    responder._generate_llm = lambda _: "You are diagnosed with depression."

    out = responder.generate(_agent_input(PolicyState.GENTLE_MONITORING))
    assert "diagnosed" not in out.lower()


def test_safe_llm_output_is_used() -> None:
    responder = GuardedLLMResponseGenerator(allow_fallback=True)
    responder._generate_llm = (
        lambda _: "Thank you for sharing this. We can talk through one small step for tonight and consider support."
    )

    out = responder.generate(_agent_input(PolicyState.NORMAL_SUPPORT))
    assert "thank you for sharing" in out.lower()


def test_visual_question_uses_visual_affect_probs() -> None:
    responder = GuardedLLMResponseGenerator(allow_fallback=True)
    data = AgentInput(
        user_text="How is my facial expression?",
        risk_score=0.20,
        policy_state=PolicyState.NORMAL_SUPPORT,
        visual_affect_probs=[0.10, 0.20, 0.70],
    )
    out = responder.generate(data)
    assert "positive/engaged" in out.lower()


def test_visual_question_high_confidence_can_use_llm() -> None:
    responder = GuardedLLMResponseGenerator(allow_fallback=True)
    responder._generate_llm = lambda _: "You look mostly positive/engaged in this clip with reasonably high confidence."
    data = AgentInput(
        user_text="How is my facial expression?",
        risk_score=0.20,
        policy_state=PolicyState.NORMAL_SUPPORT,
        visual_affect_probs=[0.05, 0.15, 0.80],
    )
    out = responder.generate(data)
    assert "positive/engaged" in out.lower()


def test_visual_question_low_signal_message() -> None:
    responder = GuardedLLMResponseGenerator(allow_fallback=True)
    data = AgentInput(
        user_text="Do I look okay in this video?",
        risk_score=0.20,
        policy_state=PolicyState.NORMAL_SUPPORT,
        visual_affect_probs=[0.34, 0.33, 0.33],
    )
    out = responder.generate(data)
    assert "confidence is low" in out.lower()
