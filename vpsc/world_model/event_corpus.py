"""Strict episode-aware LM corpus for TextWorld ``token_events.txt`` files.

Expected layout::

    root/
      train/token_events.txt
      valid/token_events.txt
      test/token_events.txt

The files are produced by ``textworld_dataset.write_token_event_text``.  Every
physical line has ``six-digit-step<TAB><|channel|><TAB>JSON`` form and each
episode starts with a line whose channel field is exactly ``<|episode|>``.
Episode header payloads must declare the same split as their containing file.

Each parsed episode receives one ``<bos>`` token and every physical event line
receives one trailing ``<eos>``.  Vocabulary construction sees only train
episodes; validation and test use the resulting ``<unk>`` ID.  Chunking is
performed independently per episode, so state reset boundaries cannot be lost
through flattening or a long sequence length.

There is deliberately no generated, synthetic, or empty-data fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Dict, Iterator, Mapping, Optional, Tuple, Union

from .wikitext import (
    LMTokenChunk,
    RegexWordPunctTokenizer,
    SPLITS,
    Vocabulary,
    file_sha256,
)


PathLike = Union[str, Path]
EVENT_FILENAME = "token_events.txt"
EPISODE_CHANNEL = "<|episode|>"
_EVENT_LINE = re.compile(
    r"(?P<step>[0-9]{6})\t(?P<channel><\|[a-z][a-z0-9_]*\|>)\t(?P<payload>.+)\Z"
)


class EventCorpusError(RuntimeError):
    """Base error for strict event-corpus loading."""


class EventCorpusNotFoundError(FileNotFoundError, EventCorpusError):
    """A required split source is missing; no fallback was attempted."""


class EventCorpusFormatError(EventCorpusError):
    """A split is empty or violates the token-event episode format."""


@dataclass(frozen=True)
class EventCorpusPaths:
    """Resolved paths of the three mandatory source files."""

    root: Path
    train: Path
    valid: Path
    test: Path

    @classmethod
    def from_root(cls, root: PathLike) -> "EventCorpusPaths":
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            train=resolved / "train" / EVENT_FILENAME,
            valid=resolved / "valid" / EVENT_FILENAME,
            test=resolved / "test" / EVENT_FILENAME,
        )

    def split(self, name: str) -> Path:
        _validate_split(name)
        return getattr(self, name)


@dataclass(frozen=True)
class EventSplitMetadata:
    """Auditable source and tokenisation counts for one split.

    ``token_count`` includes one BOS per episode and one EOS per physical line.
    """

    split: str
    source_path: str
    source_sha256: str
    source_bytes: int
    episode_count: int
    physical_line_count: int
    token_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "split": self.split,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "episode_count": self.episode_count,
            "physical_line_count": self.physical_line_count,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class TextWorldEventCorpus:
    """Immutable episode token IDs with a train-only vocabulary."""

    paths: EventCorpusPaths
    tokenizer: RegexWordPunctTokenizer
    vocabulary: Vocabulary
    _episodes: Mapping[str, Tuple[Tuple[int, ...], ...]]
    _metadata: Mapping[str, EventSplitMetadata]

    def split_metadata(self, split: str) -> EventSplitMetadata:
        _validate_split(split)
        return self._metadata[split]

    def episode_count(self, split: str) -> int:
        return self.split_metadata(split).episode_count

    def token_count(self, split: str) -> int:
        return self.split_metadata(split).token_count

    def iter_episode_token_ids(self, split: str) -> Iterator[Tuple[int, ...]]:
        """Yield complete episodes, each beginning with BOS and ending in EOS."""

        _validate_split(split)
        yield from self._episodes[split]

    def iter_chunks(
        self,
        split: str,
        sequence_length: int,
        *,
        drop_last: bool = False,
        as_tensors: bool = False,
        device: Optional[object] = None,
    ) -> Iterator[LMTokenChunk]:
        """Yield shifted chunks without ever crossing an episode boundary.

        The first yielded chunk of every episode has ``reset_state=True``;
        later chunks from that episode are contiguous and retain state.
        """

        _validate_split(split)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if device is not None and not as_tensors:
            raise ValueError("device requires as_tensors=True")

        for episode in self._episodes[split]:
            first = True
            final_input_offset = len(episode) - 1
            for offset in range(0, final_input_offset, sequence_length):
                length = min(sequence_length, final_input_offset - offset)
                if drop_last and length < sequence_length:
                    break
                inputs = episode[offset : offset + length]
                targets = episode[offset + 1 : offset + length + 1]
                yield LMTokenChunk(
                    input_ids=_maybe_tensor(inputs, as_tensors, device),
                    target_ids=_maybe_tensor(targets, as_tensors, device),
                    length=length,
                    reset_state=first,
                )
                first = False

    def metadata(self) -> Dict[str, object]:
        """Return source hashes/counts and vocabulary provenance as JSON data."""

        return {
            "format": "textworld-token-events-v1",
            "episode_channel": EPISODE_CHANNEL,
            "tokenizer": self.tokenizer.metadata(),
            "vocabulary_size": len(self.vocabulary),
            "vocabulary_fingerprint": self.vocabulary.fingerprint,
            "splits": {
                split: self._metadata[split].to_dict() for split in SPLITS
            },
        }


@dataclass(frozen=True)
class _ParsedSplit:
    token_episodes: Tuple[Tuple[str, ...], ...]
    metadata: EventSplitMetadata


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS!r}, got {split!r}")


def _strict_json(payload: str, *, path: Path, line_number: int) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant {value}")

    try:
        return json.loads(payload, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise EventCorpusFormatError(
            f"invalid JSON payload in {path} line {line_number}: {error}"
        ) from error


def _parse_source(
    path: Path,
    split: str,
    tokenizer: RegexWordPunctTokenizer,
) -> _ParsedSplit:
    if not path.is_file():
        raise EventCorpusNotFoundError(
            f"required {split} event source is missing: {path}; "
            "no synthetic fallback was used"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EventCorpusFormatError(f"cannot read UTF-8 event source {path}: {error}") from error
    if not text or not text.strip():
        raise EventCorpusFormatError(f"event source is empty: {path}")

    physical_lines = text.splitlines()
    episodes = []
    current_tokens = None
    current_line_count = 0
    previous_step = 0

    def finish_episode(next_line_number: int) -> None:
        nonlocal current_tokens, current_line_count
        if current_tokens is None:
            return
        if current_line_count <= 1:
            raise EventCorpusFormatError(
                f"episode before {path} line {next_line_number} has no event records"
            )
        episodes.append(tuple(current_tokens))
        current_tokens = None
        current_line_count = 0

    for line_number, line in enumerate(physical_lines, start=1):
        if not line or not line.strip():
            raise EventCorpusFormatError(f"blank physical line in {path} line {line_number}")
        match = _EVENT_LINE.fullmatch(line)
        if match is None:
            raise EventCorpusFormatError(
                f"malformed event line in {path} line {line_number}; expected "
                "six-digit-step<TAB><|channel|><TAB>JSON"
            )
        step = int(match.group("step"))
        channel = match.group("channel")
        payload = _strict_json(match.group("payload"), path=path, line_number=line_number)

        if channel == EPISODE_CHANNEL:
            finish_episode(line_number)
            if step != 0:
                raise EventCorpusFormatError(
                    f"episode marker must use step 000000 in {path} line {line_number}"
                )
            if not isinstance(payload, dict) or payload.get("split") != split:
                raise EventCorpusFormatError(
                    f"episode header split must be {split!r} in {path} line {line_number}"
                )
            current_tokens = ["<bos>"]
            current_line_count = 0
            previous_step = 0
        elif current_tokens is None:
            raise EventCorpusFormatError(
                f"event before first {EPISODE_CHANNEL} marker in {path} line {line_number}"
            )
        elif step < previous_step:
            raise EventCorpusFormatError(
                f"event step decreased inside episode in {path} line {line_number}"
            )

        assert current_tokens is not None
        current_tokens.extend(tokenizer.tokenize(line))
        current_tokens.append("<eos>")
        current_line_count += 1
        previous_step = step

    finish_episode(len(physical_lines) + 1)
    if not episodes:
        raise EventCorpusFormatError(f"event source contains no episodes: {path}")

    token_count = sum(len(episode) for episode in episodes)
    metadata = EventSplitMetadata(
        split=split,
        source_path=str(path.resolve()),
        source_sha256=file_sha256(path),
        source_bytes=path.stat().st_size,
        episode_count=len(episodes),
        physical_line_count=len(physical_lines),
        token_count=token_count,
    )
    return _ParsedSplit(token_episodes=tuple(episodes), metadata=metadata)


def _maybe_tensor(
    values: Tuple[int, ...],
    as_tensors: bool,
    device: Optional[object],
) -> object:
    if not as_tensors:
        return values
    try:
        import torch
    except ImportError as error:  # pragma: no cover - only in torch-free runtime
        raise ImportError("PyTorch is required when as_tensors=True") from error
    return torch.tensor(values, dtype=torch.long, device=device)


def load_event_corpus(
    root: PathLike,
    *,
    lowercase: bool = False,
    min_frequency: int = 1,
    max_vocab_size: Optional[int] = None,
) -> TextWorldEventCorpus:
    """Load and validate all splits, then build a train-only vocabulary."""

    paths = EventCorpusPaths.from_root(root)
    source_paths = tuple(paths.split(split) for split in SPLITS)
    if len({str(path.resolve()) for path in source_paths}) != len(SPLITS):
        raise EventCorpusFormatError("train, valid, and test must be distinct files")

    tokenizer = RegexWordPunctTokenizer(lowercase=lowercase)
    parsed = {
        split: _parse_source(paths.split(split), split, tokenizer) for split in SPLITS
    }
    vocabulary = Vocabulary.build(
        (
            token
            for episode in parsed["train"].token_episodes
            for token in episode
        ),
        min_frequency=min_frequency,
        max_size=max_vocab_size,
    )
    encoded = {
        split: tuple(
            vocabulary.encode(episode) for episode in parsed[split].token_episodes
        )
        for split in SPLITS
    }
    metadata = {split: parsed[split].metadata for split in SPLITS}
    return TextWorldEventCorpus(
        paths=paths,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        _episodes=MappingProxyType(encoded),
        _metadata=MappingProxyType(metadata),
    )


load_textworld_event_corpus = load_event_corpus


__all__ = [
    "EPISODE_CHANNEL",
    "EVENT_FILENAME",
    "EventCorpusError",
    "EventCorpusFormatError",
    "EventCorpusNotFoundError",
    "EventCorpusPaths",
    "EventSplitMetadata",
    "TextWorldEventCorpus",
    "load_event_corpus",
    "load_textworld_event_corpus",
]
