from __future__ import annotations

from scripts.red_team import (
    SimulatedTargetEndpoint,
    build_audit_record,
    generate_attack_cases,
    summarize_batch,
)


def test_generate_attack_cases_builds_prompt_injection_variants() -> None:
    attacks = generate_attack_cases(count=3, seed=7)

    assert len(attacks) == 3
    assert all(attack.expected_labels == ("jailbreak",) for attack in attacks)
    assert {attack.category for attack in attacks} == {"prompt-injection", "jailbreak"}
    assert all(attack.prompt for attack in attacks)


def test_simulated_target_response_can_be_serialized_as_audit_record() -> None:
    attack = generate_attack_cases(count=1, seed=2)[0]
    response = SimulatedTargetEndpoint().generate(attack)

    record = build_audit_record(
        response=response,
        run_id="redteam-test",
        project_name="test-project",
        model_name="simulated-model",
        user_id="tester",
        extra_tags=["ci"],
    )

    assert record["input_text"] == attack.prompt
    assert "ignore previous instructions" in str(record["output_text"]).lower()
    assert record["project_name"] == "test-project"
    assert "red-team" in record["tags"]
    assert "ci" in record["tags"]


def test_summarize_batch_reports_detection_success_rate_inputs() -> None:
    attacks = generate_attack_cases(count=2, seed=3)
    payload = {
        "count": 2,
        "results": [
            {"flagged": True, "risk_labels": ["jailbreak"], "jailbreak": True},
            {"flagged": False, "risk_labels": [], "jailbreak": False},
        ],
    }

    stats = summarize_batch(attacks, payload)

    assert stats.attempted == 2
    assert stats.detected == 1
    assert stats.flagged == 1
