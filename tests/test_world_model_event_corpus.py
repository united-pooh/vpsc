import json
from pathlib import Path
import tempfile
import unittest

from vpsc.world_model.event_corpus import (
    EPISODE_CHANNEL,
    EventCorpusFormatError,
    EventCorpusNotFoundError,
    load_event_corpus,
)
from vpsc.world_model.wikitext import LMTokenChunk, file_sha256


def _line(step: int, channel: str, payload: object) -> str:
    return (
        f"{step:06d}\t<|{channel}|>\t"
        f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    )


def _episode(split: str, seed: int, unique_word: str) -> list[str]:
    return [
        _line(0, "episode", {"seed": seed, "split": split}),
        _line(0, "observation", f"room contains {unique_word}"),
        _line(0, "action", f"take {unique_word}"),
        _line(1, "end_episode", True),
    ]


def _write_split(root: Path, split: str, episodes: list[list[str]]) -> Path:
    path = root / split / "token_events.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(line for episode in episodes for line in episode) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _write_complete_fixture(root: Path) -> dict[str, Path]:
    return {
        "train": _write_split(
            root,
            "train",
            [
                _episode("train", 11, "trainalpha"),
                _episode("train", 12, "trainbeta"),
            ],
        ),
        "valid": _write_split(
            root,
            "valid",
            [_episode("valid", 21, "validviolet")],
        ),
        "test": _write_split(
            root,
            "test",
            [_episode("test", 31, "testamber")],
        ),
    }


class TextWorldEventCorpusTests(unittest.TestCase):
    def test_train_only_vocabulary_oov_and_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _write_complete_fixture(root)
            corpus = load_event_corpus(root)

            self.assertIn("trainalpha", corpus.vocabulary)
            self.assertIn("trainbeta", corpus.vocabulary)
            self.assertNotIn("validviolet", corpus.vocabulary)
            self.assertNotIn("testamber", corpus.vocabulary)

            train_episodes = list(corpus.iter_episode_token_ids("train"))
            valid_episode = next(corpus.iter_episode_token_ids("valid"))
            test_episode = next(corpus.iter_episode_token_ids("test"))
            self.assertTrue(
                all(corpus.vocabulary.unk_id not in episode for episode in train_episodes)
            )
            self.assertIn(corpus.vocabulary.unk_id, valid_episode)
            self.assertIn(corpus.vocabulary.unk_id, test_episode)

            for episode in train_episodes:
                self.assertEqual(episode[0], corpus.vocabulary.bos_id)
                self.assertEqual(episode[-1], corpus.vocabulary.eos_id)
                # Four physical source lines in every fixture episode.
                self.assertEqual(episode.count(corpus.vocabulary.eos_id), 4)

            train_metadata = corpus.split_metadata("train")
            self.assertEqual(train_metadata.source_sha256, file_sha256(paths["train"]))
            self.assertEqual(train_metadata.source_bytes, paths["train"].stat().st_size)
            self.assertEqual(train_metadata.episode_count, 2)
            self.assertEqual(train_metadata.physical_line_count, 8)
            self.assertEqual(
                train_metadata.token_count,
                sum(len(episode) for episode in train_episodes),
            )
            metadata = corpus.metadata()
            self.assertEqual(metadata["episode_channel"], EPISODE_CHANNEL)
            json.dumps(metadata, allow_nan=False)

    def test_chunks_reconstruct_each_episode_without_crossing_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_complete_fixture(root)
            corpus = load_event_corpus(root)
            expected_episodes = list(corpus.iter_episode_token_ids("train"))
            chunks = list(corpus.iter_chunks("train", sequence_length=7))

            self.assertTrue(all(isinstance(chunk, LMTokenChunk) for chunk in chunks))
            self.assertEqual(sum(chunk.reset_state for chunk in chunks), 2)
            self.assertTrue(chunks[0].reset_state)

            reconstructed = []
            current = None
            previous_target = None
            for chunk in chunks:
                inputs = tuple(chunk.input_ids)
                targets = tuple(chunk.target_ids)
                self.assertEqual(len(inputs), chunk.length)
                self.assertEqual(len(targets), chunk.length)
                if chunk.reset_state:
                    if current is not None:
                        reconstructed.append(tuple(current))
                    current = list(inputs) + [targets[-1]]
                else:
                    self.assertIsNotNone(current)
                    self.assertEqual(previous_target, inputs[0])
                    current.extend(targets)
                previous_target = targets[-1]
            reconstructed.append(tuple(current))

            self.assertEqual(reconstructed, expected_episodes)

            one_chunk_per_episode = list(
                corpus.iter_chunks("train", sequence_length=10_000)
            )
            self.assertEqual(len(one_chunk_per_episode), 2)
            self.assertTrue(all(chunk.reset_state for chunk in one_chunk_per_episode))

    def test_missing_split_is_a_hard_failure_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_split(root, "train", [_episode("train", 1, "alpha")])
            _write_split(root, "valid", [_episode("valid", 2, "beta")])

            with self.assertRaisesRegex(EventCorpusNotFoundError, "no synthetic fallback"):
                load_event_corpus(root)
            self.assertFalse((root / "test" / "token_events.txt").exists())

    def test_empty_split_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _write_complete_fixture(root)
            paths["valid"].write_text("", encoding="utf-8")

            with self.assertRaisesRegex(EventCorpusFormatError, "empty"):
                load_event_corpus(root)

    def test_malformed_or_empty_episode_is_a_hard_failure(self) -> None:
        malformed_cases = {
            "event before marker": _line(0, "observation", "orphan") + "\n",
            "empty episode": "\n".join(
                (
                    _line(0, "episode", {"seed": 1, "split": "train"}),
                    _line(0, "episode", {"seed": 2, "split": "train"}),
                    _line(0, "observation", "second episode"),
                )
            )
            + "\n",
            "wrong header split": "\n".join(
                (
                    _line(0, "episode", {"seed": 1, "split": "valid"}),
                    _line(0, "observation", "wrong split"),
                )
            )
            + "\n",
        }
        for name, malformed in malformed_cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = _write_complete_fixture(root)
                paths["train"].write_text(malformed, encoding="utf-8", newline="\n")
                with self.assertRaises(EventCorpusFormatError):
                    load_event_corpus(root)


if __name__ == "__main__":
    unittest.main()
