from pathlib import Path

from experiments.e3_sg1_history_generation import (
    audit_history_examples,
    build_history_examples,
)
from experiments.e3_sg1_history_identifiability import (
    build_identifiability_records,
    extract_room_name,
    run_audit,
)
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e2_world_model/textworld_l5")
EXPECTED_VOCABULARY_FINGERPRINT = (
    "2480444d34a970d03f8e0b1c59643b432012769f9362d9eda402943ec827c314"
)


def _contains_subsequence(sequence: tuple[int, ...], query: tuple[int, ...]) -> bool:
    return any(
        sequence[index : index + len(query)] == query
        for index in range(len(sequence) - len(query) + 1)
    )


def test_room_name_extraction_is_strict_and_normalized() -> None:
    assert extract_room_name("-= Spare Room =-\nA room.") == "Spare Room"
    assert extract_room_name("You are carrying nothing.") is None
    assert extract_room_name("-= =-\nEmpty header.") is None


def test_sg1_identifiability_audit_finds_history_not_current() -> None:
    records = build_identifiability_records(CORPUS_ROOT)
    assert {
        split: sum(record.action_type == "move" for record in values)
        for split, values in records.items()
    } == {"train": 16, "valid": 4, "test": 4}
    for values in records.values():
        for record in values:
            if record.action_type == "move":
                assert not record.target_room_visible_in_current
                assert record.target_room_in_prior_history
                assert record.target_surface_in_prior_history
                assert record.target_surface_history_lag == 1

    audit = run_audit(CORPUS_ROOT)
    assert audit["decision"] == {
        "sg0_single_observation_identifiability": "FAIL",
        "sg1_history_conditioned_route": "PASS",
        "next_experiment": "history_conditioned_generation",
    }
    assert audit["held_out_move"]["target_room_visible_in_current_ratio"] == 0
    assert audit["held_out_move"]["target_surface_in_prior_history_ratio"] == 1


def test_sg1_history_examples_are_deterministic_and_targets_are_visible() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    first, first_vocabulary, first_previous = build_history_examples(
        CORPUS_ROOT, corpus
    )
    second, second_vocabulary, second_previous = build_history_examples(
        CORPUS_ROOT, corpus
    )
    assert first == second
    assert first_previous == second_previous
    assert first_vocabulary.tokens == second_vocabulary.tokens
    assert len(first_vocabulary) == 186
    assert first_vocabulary.fingerprint == EXPECTED_VOCABULARY_FINGERPRINT

    for split in ("train", "valid", "test"):
        for example in first[split]:
            if example.action_type == "move":
                assert _contains_subsequence(
                    example.prompt_ids, example.target_ids[:-1]
                )

    audit = audit_history_examples(
        CORPUS_ROOT, first, first_vocabulary, first_previous
    )
    assert audit["passed"]
    assert audit["held_out_move_target_surface_in_history"]["ratio"] == 1
    assert audit["splits"]["train"]["prompt_length"]["max"] == 301
    assert audit["splits"]["test"]["prompt_length"]["max"] == 284
    assert audit["baselines"]["history_rule"]["exact"] == 0.9
    assert audit["baselines"]["history_rule"]["action_types"]["move"][
        "room_accuracy"
    ] == 1.0
