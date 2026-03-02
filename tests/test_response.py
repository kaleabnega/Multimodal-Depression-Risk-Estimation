from mde.core.types import AgentInput, PolicyState
from mde.services.response import GuardedLLMResponseGenerator


def _agent_input(state: PolicyState) -> AgentInput:
    return AgentInput(
        user_text="I feel very low today",
        risk_score=0.55,
        policy_state=state,
        audio_summary=None,
        visual_summary=None,
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
