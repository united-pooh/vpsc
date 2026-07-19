from __future__ import annotations

import torch

from experiments.e3_sg21r_sixth_fresh_matched_ann import (
    MatchedLSTM,
    MatchedTransformer,
    build_feature_vocabulary,
    encode_feature_tokens,
    feature_tokens,
)


ACTIONS = (
    "examine coin",
    "go east",
    "go north",
    "go south",
    "go west",
    "inventory",
    "look",
    "take coin",
)


def test_matched_feature_prompt_has_frozen_sixteen_slots() -> None:
    plan = ("go west", "go north", "go east", "go south", "take coin")
    mask = (0, 1, 0, 1, 0, 1, 1, 0)
    tokens = feature_tokens(
        ("go west", "go north"),
        mask,
        "go south",
        plan,
        "go north",
        ACTIONS,
    )

    assert len(tokens) == 16
    assert tokens[0] == "phase:2"
    assert tokens[1:4] == (
        "history:0:<pad>",
        "history:1:go west",
        "history:2:go north",
    )
    assert "candidate:go south" in tokens
    assert "return:1" in tokens
    assert tokens[-8:] == tuple(
        f"mask:{action}:{bit}" for action, bit in zip(ACTIONS, mask)
    )


def test_feature_vocabulary_encodes_runtime_prompt_without_oov() -> None:
    vocabulary = build_feature_vocabulary(ACTIONS, ACTIONS)
    tokens = feature_tokens(
        ("go west", "go north", "go east", "go south"),
        (0, 1, 0, 0, 1, 1, 1, 1),
        "take coin",
        ("go west", "go north", "go east", "go south", "take coin"),
        "go south",
        ACTIONS,
    )

    encoded = encode_feature_tokens(tokens, vocabulary)
    assert len(encoded) == 16
    assert all(0 <= value < len(vocabulary.tokens) for value in encoded)


def test_matched_ann_models_emit_nineteen_world_state_logits() -> None:
    vocabulary = build_feature_vocabulary(ACTIONS, ACTIONS)
    token_ids = torch.zeros((3, 16), dtype=torch.long)

    lstm = MatchedLSTM(len(vocabulary.tokens))
    transformer = MatchedTransformer(len(vocabulary.tokens))

    assert lstm(token_ids).shape == (3, 19)
    assert transformer(token_ids).shape == (3, 19)
