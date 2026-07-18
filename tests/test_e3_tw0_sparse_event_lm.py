from pathlib import Path

from experiments.e3_tw0_sparse_event_lm import (
    SELECTED_CHANNELS,
    _line_payload_channels,
    audit_sparse_chunks,
    build_sparse_chunks,
)
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e2_world_model/textworld_l5")


def test_real_textworld_sparse_chunks_are_deterministic_and_valid() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    first = {
        split: build_sparse_chunks(corpus, split)
        for split in ("train", "valid", "test")
    }
    second = {
        split: build_sparse_chunks(corpus, split)
        for split in ("train", "valid", "test")
    }
    assert first == second
    audit = audit_sparse_chunks(first, corpus.vocabulary)
    assert audit["passed"]
    assert audit["format_token_ratio"] < 0.70
    assert {
        split: audit["splits"][split]["sparse_query_count"]
        for split in ("train", "valid", "test")
    } == {"train": 384, "valid": 96, "test": 96}


def test_sparse_queries_stay_inside_selected_payload_channels() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    chunks = build_sparse_chunks(corpus, "train")
    episodes = tuple(corpus.iter_episode_token_ids("train"))
    channel_maps = tuple(
        _line_payload_channels(episode, corpus.vocabulary) for episode in episodes
    )
    reset_count = 0
    for chunk in chunks:
        reset_count += int(chunk.reset_state)
        assert len(chunk.sparse_query_indices) <= 16
        assert len(chunk.sparse_query_indices) == len(chunk.sparse_query_channels)
        assert set(chunk.sparse_query_channels) <= SELECTED_CHANNELS
        assert all(
            0 <= index < len(chunk.target_ids)
            for index in chunk.sparse_query_indices
        )
        assert all(
            right > left
            for left, right in zip(
                chunk.sparse_query_indices, chunk.sparse_query_indices[1:]
            )
        )
        for index, channel in zip(
            chunk.sparse_query_indices, chunk.sparse_query_channels
        ):
            global_target_position = chunk.offset + index + 1
            assert channel_maps[chunk.episode_index][global_target_position] == channel
    assert reset_count == corpus.episode_count("train") == 4
