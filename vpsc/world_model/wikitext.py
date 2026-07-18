"""Reproducible WikiText-2 raw data access for causal language modelling.

The adapter deliberately keeps acquisition separate from modelling:

* network access happens only when ``download=True`` is passed explicitly;
* the canonical archive is checked against a pinned SHA256 before extraction;
* cached split files are checked against a manifest derived from that archive;
* the vocabulary is built from the training split only; and
* train, validation, and test streams are never concatenated.

There is no synthetic-data fallback.  Missing, unreachable, or corrupt data
raises a specific exception so an experiment cannot silently change datasets.
Only the Python standard library is required.  PyTorch is imported lazily when
``as_tensors=True`` is requested from a chunk or batch iterator.
"""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


PathLike = Union[str, os.PathLike]
SPLITS: Tuple[str, ...] = ("train", "valid", "test")
SPECIAL_TOKENS: Tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")
WORD_PUNCT_PATTERN = r"\w+(?:['’]\w+)*|[^\w\s]"
_HEX_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_SAFE_CACHE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class WikiTextError(RuntimeError):
    """Base class for errors raised by this adapter."""


class WikiTextNotFoundError(FileNotFoundError, WikiTextError):
    """The verified archive is not cached and download was not requested."""


class WikiTextDownloadError(WikiTextError):
    """The explicitly requested archive download failed."""


class WikiTextIntegrityError(WikiTextError):
    """An archive or extracted split failed an integrity check."""


@dataclass(frozen=True)
class WikiText2Source:
    """Pinned source metadata for a WikiText-2-compatible raw archive.

    ``split_members`` is a tuple rather than a mutable mapping so the source
    description itself is deterministic and hashable.  Supplying a custom
    source makes it possible to exercise the complete download/check/extract
    path with a tiny local ``file://`` fixture and no network access.
    """

    name: str
    version: str
    url: str
    sha256: str
    archive_filename: str
    archive_bytes: Optional[int]
    split_members: Tuple[Tuple[str, str], ...]
    homepage: str = ""
    license: str = ""

    def __post_init__(self) -> None:
        if not _SAFE_CACHE_NAME.fullmatch(self.name):
            raise ValueError(f"unsafe source name: {self.name!r}")
        if Path(self.archive_filename).name != self.archive_filename:
            raise ValueError("archive_filename must be a plain file name")
        if not _HEX_SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        if self.archive_bytes is not None and self.archive_bytes <= 0:
            raise ValueError("archive_bytes must be positive when provided")

        members = dict(self.split_members)
        if len(members) != len(self.split_members):
            raise ValueError("split_members contains duplicate split names")
        if tuple(members) != SPLITS:
            raise ValueError(f"split_members must be ordered exactly as {SPLITS!r}")
        for member in members.values():
            pure = Path(member.replace("\\", "/"))
            if pure.is_absolute() or ".." in pure.parts or not pure.name:
                raise ValueError(f"unsafe archive member: {member!r}")

    def member_for(self, split: str) -> str:
        _validate_split(split)
        return dict(self.split_members)[split]

    def metadata(self) -> Dict[str, object]:
        """Return JSON-serialisable provenance metadata."""

        return {
            "name": self.name,
            "version": self.version,
            "url": self.url,
            "sha256": self.sha256.lower(),
            "archive_filename": self.archive_filename,
            "archive_bytes": self.archive_bytes,
            "split_members": [list(item) for item in self.split_members],
            "homepage": self.homepage,
            "license": self.license,
        }


WIKITEXT2_RAW_SOURCE = WikiText2Source(
    name="wikitext-2-raw-v1",
    version="1.0.0",
    # The original MetaMind S3 endpoint now returns an unusable HTTP 301.
    # ggml-org mirrors the exact Git-LFS object: its byte size and SHA-256
    # match Salesforce's frozen metadata below, which remains authoritative.
    url=(
        "https://huggingface.co/datasets/ggml-org/ci/resolve/main/"
        "wikitext-2-raw-v1.zip"
    ),
    sha256="ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
    archive_filename="wikitext-2-raw-v1.zip",
    archive_bytes=4_721_645,
    split_members=(
        ("train", "wikitext-2-raw/wiki.train.raw"),
        ("valid", "wikitext-2-raw/wiki.valid.raw"),
        ("test", "wikitext-2-raw/wiki.test.raw"),
    ),
    homepage=(
        "https://blog.einstein.ai/"
        "the-wikitext-long-term-dependency-language-modeling-dataset/"
    ),
    license="Creative Commons Attribution-ShareAlike 4.0 International",
)


@dataclass(frozen=True)
class WikiText2Paths:
    """Locations of one verified WikiText-2 cache."""

    root: Path
    archive: Path
    manifest: Path
    train: Path
    valid: Path
    test: Path

    def split(self, name: str) -> Path:
        _validate_split(name)
        return getattr(self, name)


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS!r}, got {split!r}")


def file_sha256(path: PathLike, block_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA256 digest of *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _cache_paths(cache_dir: PathLike, source: WikiText2Source) -> WikiText2Paths:
    root = Path(cache_dir).expanduser().resolve() / source.name
    splits_dir = root / "splits"
    split_paths = {
        split: splits_dir / Path(source.member_for(split)).name for split in SPLITS
    }
    if len(set(split_paths.values())) != len(SPLITS):
        raise ValueError("split archive members must have distinct file names")
    return WikiText2Paths(
        root=root,
        archive=root / source.archive_filename,
        manifest=root / "manifest.json",
        train=split_paths["train"],
        valid=split_paths["valid"],
        test=split_paths["test"],
    )


def verify_wikitext2_archive(
    archive: PathLike,
    source: WikiText2Source = WIKITEXT2_RAW_SOURCE,
) -> str:
    """Verify archive byte length (when pinned) and SHA256, then return SHA256."""

    path = Path(archive)
    if not path.is_file():
        raise WikiTextNotFoundError(f"WikiText archive is missing: {path}")
    actual_bytes = path.stat().st_size
    if source.archive_bytes is not None and actual_bytes != source.archive_bytes:
        raise WikiTextIntegrityError(
            f"WikiText archive size mismatch for {path}: expected "
            f"{source.archive_bytes}, got {actual_bytes}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != source.sha256.lower():
        raise WikiTextIntegrityError(
            f"WikiText archive SHA256 mismatch for {path}: expected "
            f"{source.sha256.lower()}, got {actual_sha256}"
        )
    return actual_sha256


def download_wikitext2(
    cache_dir: PathLike,
    *,
    source: WikiText2Source = WIKITEXT2_RAW_SOURCE,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Explicitly download and verify the pinned archive into *cache_dir*.

    A corrupt existing archive is never silently replaced.  Pass ``force=True``
    to state that replacement intent explicitly.  The download is written to a
    temporary sibling and moved into place only after its SHA256 has passed.
    """

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    paths = _cache_paths(cache_dir, source)
    if paths.archive.exists() and not force:
        verify_wikitext2_archive(paths.archive, source)
        return paths.archive

    paths.root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{source.archive_filename}.", suffix=".part", dir=str(paths.root)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_count = 0
    request = Request(source.url, headers={"User-Agent": "vpsc-wikitext/1.0"})
    try:
        try:
            with urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    byte_count += len(block)
        except (HTTPError, URLError, OSError) as error:
            raise WikiTextDownloadError(
                f"failed to download WikiText archive from {source.url}: {error}"
            ) from error

        actual_sha256 = digest.hexdigest()
        if source.archive_bytes is not None and byte_count != source.archive_bytes:
            raise WikiTextIntegrityError(
                f"downloaded WikiText archive size mismatch: expected "
                f"{source.archive_bytes}, got {byte_count}"
            )
        if actual_sha256 != source.sha256.lower():
            raise WikiTextIntegrityError(
                f"downloaded WikiText archive SHA256 mismatch: expected "
                f"{source.sha256.lower()}, got {actual_sha256}"
            )
        os.replace(temporary, paths.archive)
    finally:
        temporary.unlink(missing_ok=True)

    return paths.archive


def _manifest_payload(
    source: WikiText2Source,
    split_metadata: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    return {
        "format_version": 1,
        "dataset": "WikiText-2 raw",
        "source": source.metadata(),
        "archive": {
            "filename": source.archive_filename,
            "sha256": source.sha256.lower(),
            "bytes": source.archive_bytes,
        },
        "splits": {split: dict(split_metadata[split]) for split in SPLITS},
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cached_splits_are_valid(
    paths: WikiText2Paths,
    source: WikiText2Source,
) -> bool:
    if not paths.manifest.is_file():
        return False
    try:
        with paths.manifest.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False

    if manifest.get("format_version") != 1:
        return False
    if manifest.get("source") != source.metadata():
        return False
    archive_meta = manifest.get("archive")
    if not isinstance(archive_meta, dict):
        return False
    if archive_meta.get("sha256") != source.sha256.lower():
        return False

    split_meta = manifest.get("splits")
    if not isinstance(split_meta, dict):
        return False
    for split in SPLITS:
        metadata = split_meta.get(split)
        path = paths.split(split)
        if not isinstance(metadata, dict) or not path.is_file():
            return False
        if metadata.get("filename") != path.name:
            return False
        if metadata.get("bytes") != path.stat().st_size:
            return False
        if metadata.get("sha256") != file_sha256(path):
            return False
    return True


def _extract_verified_splits(
    paths: WikiText2Paths,
    source: WikiText2Source,
) -> None:
    """Extract required members without trusting paths stored inside the zip."""

    paths.train.parent.mkdir(parents=True, exist_ok=True)
    temporary_files: Dict[str, Path] = {}
    split_metadata: Dict[str, Dict[str, object]] = {}
    try:
        try:
            with zipfile.ZipFile(paths.archive, "r") as archive:
                infos = {}
                for split in SPLITS:
                    member = source.member_for(split)
                    try:
                        info = archive.getinfo(member)
                    except KeyError as error:
                        raise WikiTextIntegrityError(
                            f"verified archive does not contain required member {member!r}"
                        ) from error
                    if info.is_dir():
                        raise WikiTextIntegrityError(
                            f"required archive member is a directory: {member!r}"
                        )
                    infos[split] = info

                for split in SPLITS:
                    target = paths.split(split)
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
                    )
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    temporary_files[split] = temporary
                    with archive.open(infos[split], "r") as source_file, temporary.open(
                        "wb"
                    ) as destination:
                        shutil.copyfileobj(source_file, destination, length=1024 * 1024)
                    split_metadata[split] = {
                        "filename": target.name,
                        "archive_member": source.member_for(split),
                        "bytes": temporary.stat().st_size,
                        "sha256": file_sha256(temporary),
                    }
        except zipfile.BadZipFile as error:
            raise WikiTextIntegrityError(
                f"verified WikiText archive is not a readable zip: {paths.archive}"
            ) from error

        for split in SPLITS:
            os.replace(temporary_files[split], paths.split(split))
        _write_json_atomic(paths.manifest, _manifest_payload(source, split_metadata))
    finally:
        for temporary in temporary_files.values():
            temporary.unlink(missing_ok=True)


def prepare_wikitext2(
    cache_dir: PathLike,
    *,
    download: bool = False,
    force_download: bool = False,
    source: WikiText2Source = WIKITEXT2_RAW_SOURCE,
    timeout: float = 60.0,
) -> WikiText2Paths:
    """Return verified split paths, optionally performing an explicit download.

    ``download=False`` guarantees no network call.  A present, verified archive
    may still be extracted or used to repair a corrupt extracted split.  A
    corrupt archive always raises unless replacement was explicitly requested
    with both ``download=True`` and ``force_download=True``.
    """

    if force_download and not download:
        raise ValueError("force_download=True requires download=True")
    paths = _cache_paths(cache_dir, source)

    if force_download:
        download_wikitext2(cache_dir, source=source, force=True, timeout=timeout)
    elif not paths.archive.is_file():
        if not download:
            raise WikiTextNotFoundError(
                f"WikiText-2 raw is not cached at {paths.archive}. Call "
                "prepare_wikitext2(..., download=True) explicitly to fetch it."
            )
        download_wikitext2(cache_dir, source=source, timeout=timeout)

    verify_wikitext2_archive(paths.archive, source)
    if not _cached_splits_are_valid(paths, source):
        _extract_verified_splits(paths, source)
    if not _cached_splits_are_valid(paths, source):
        raise WikiTextIntegrityError("extracted WikiText cache failed verification")
    return paths


@dataclass(frozen=True)
class RegexWordPunctTokenizer:
    """Deterministic Unicode word-plus-punctuation tokenizer.

    Words (including contractions) stay together and every non-whitespace,
    non-word character becomes its own token.  Physical line endings are added
    as ``<eos>`` by :meth:`iter_file_tokens`; the regex itself never invents
    split or document boundaries.
    """

    lowercase: bool = False
    pattern: str = WORD_PUNCT_PATTERN
    _compiled: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", re.compile(self.pattern, re.UNICODE))

    def tokenize(self, text: str) -> Tuple[str, ...]:
        if self.lowercase:
            text = text.lower()
        return tuple(self._compiled.findall(text))

    def iter_file_tokens(
        self,
        path: PathLike,
        *,
        add_bos: bool = True,
        add_line_eos: bool = True,
    ) -> Iterator[str]:
        if add_bos:
            yield "<bos>"
        with Path(path).open("r", encoding="utf-8", newline=None) as handle:
            for line in handle:
                yield from self.tokenize(line.rstrip("\r\n"))
                if add_line_eos:
                    yield "<eos>"

    def metadata(self) -> Dict[str, object]:
        return {
            "kind": "regex-word-punct",
            "pattern": self.pattern,
            "unicode": True,
            "lowercase": self.lowercase,
            "line_boundary_token": "<eos>",
        }


@dataclass(frozen=True)
class Vocabulary:
    """Immutable, deterministically ordered token vocabulary."""

    tokens: Tuple[str, ...]
    _token_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(f"vocabulary must begin with {SPECIAL_TOKENS!r}")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")
        mapping = MappingProxyType({token: index for index, token in enumerate(self.tokens)})
        object.__setattr__(self, "_token_to_id", mapping)

    @classmethod
    def build(
        cls,
        train_tokens: Iterable[str],
        *,
        min_frequency: int = 1,
        max_size: Optional[int] = None,
    ) -> "Vocabulary":
        """Build from training tokens, sorting by ``(-frequency, token)``.

        ``max_size`` is the total size including the four fixed special tokens.
        Equal-frequency tokens use Unicode code-point order, so input iteration
        order and platform hash randomisation cannot alter token IDs.
        """

        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if max_size is not None and max_size < len(SPECIAL_TOKENS):
            raise ValueError(
                f"max_size must be at least {len(SPECIAL_TOKENS)} for special tokens"
            )
        counts = Counter(train_tokens)
        for special in SPECIAL_TOKENS:
            counts.pop(special, None)
        ordinary = [
            (token, frequency)
            for token, frequency in counts.items()
            if frequency >= min_frequency
        ]
        ordinary.sort(key=lambda item: (-item[1], item[0]))
        if max_size is not None:
            ordinary = ordinary[: max_size - len(SPECIAL_TOKENS)]
        return cls(SPECIAL_TOKENS + tuple(token for token, _ in ordinary))

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def token_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_id)

    def encode(self, tokens: Iterable[str]) -> Tuple[int, ...]:
        return tuple(self.token_id(token) for token in tokens)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special: bool = False,
    ) -> Tuple[str, ...]:
        decoded = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.tokens):
                raise IndexError(f"token id is outside vocabulary: {token_id}")
            token = self.tokens[token_id]
            if not skip_special or token not in SPECIAL_TOKENS:
                decoded.append(token)
        return tuple(decoded)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.tokens, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LMTokenChunk:
    """One contiguous, shifted next-token training chunk."""

    input_ids: object
    target_ids: object
    length: int
    reset_state: bool


@dataclass(frozen=True)
class LMTokenBatch:
    """Parallel contiguous streams suitable for stateful truncated BPTT."""

    input_ids: object
    target_ids: object
    batch_size: int
    sequence_length: int
    reset_state: bool


def _maybe_tensor(data: object, as_tensors: bool, device: Optional[object]) -> object:
    if not as_tensors:
        if device is not None:
            raise ValueError("device requires as_tensors=True")
        return data
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised where torch is absent
        raise ImportError(
            "PyTorch is required only when as_tensors=True; install torch or "
            "request tuple output"
        ) from error
    return torch.tensor(data, dtype=torch.long, device=device)


@dataclass(frozen=True)
class WikiText2Corpus:
    """Lazy split streams sharing one training-only vocabulary."""

    paths: WikiText2Paths
    tokenizer: RegexWordPunctTokenizer
    vocabulary: Vocabulary

    def iter_tokens(self, split: str) -> Iterator[str]:
        """Yield one split beginning with ``<bos>`` and ending lines with ``<eos>``."""

        return self.tokenizer.iter_file_tokens(self.paths.split(split))

    def iter_token_ids(self, split: str) -> Iterator[int]:
        for token in self.iter_tokens(split):
            yield self.vocabulary.token_id(token)

    def token_count(self, split: str) -> int:
        return sum(1 for _ in self.iter_token_ids(split))

    def encode_text(self, text: str) -> Tuple[int, ...]:
        return self.vocabulary.encode(self.tokenizer.tokenize(text))

    def iter_chunks(
        self,
        split: str,
        sequence_length: int,
        *,
        drop_last: bool = True,
        as_tensors: bool = False,
        device: Optional[object] = None,
    ) -> Iterator[LMTokenChunk]:
        """Yield non-overlapping contiguous LM chunks without crossing a split.

        The last target of one full chunk becomes the first input of the next,
        so carrying recurrent state between chunks loses no token.  Consumers
        should reset model state when ``reset_state`` is true.
        """

        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        buffer = []
        first = True
        for token_id in self.iter_token_ids(split):
            buffer.append(token_id)
            if len(buffer) == sequence_length + 1:
                inputs = tuple(buffer[:-1])
                targets = tuple(buffer[1:])
                yield LMTokenChunk(
                    input_ids=_maybe_tensor(inputs, as_tensors, device),
                    target_ids=_maybe_tensor(targets, as_tensors, device),
                    length=sequence_length,
                    reset_state=first,
                )
                first = False
                buffer = buffer[-1:]
        if not drop_last and len(buffer) >= 2:
            inputs = tuple(buffer[:-1])
            targets = tuple(buffer[1:])
            yield LMTokenChunk(
                input_ids=_maybe_tensor(inputs, as_tensors, device),
                target_ids=_maybe_tensor(targets, as_tensors, device),
                length=len(inputs),
                reset_state=first,
            )

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        sequence_length: int,
        *,
        drop_last: bool = True,
        as_tensors: bool = False,
        device: Optional[object] = None,
    ) -> Iterator[LMTokenBatch]:
        """Yield standard batchified, contiguous streams for truncated BPTT.

        The split is divided into ``batch_size`` long lanes.  Successive yielded
        batches are contiguous within every lane, which permits persistent LSTM
        or E2 state.  Tokens that cannot fill all lanes are deterministically
        discarded from the tail.  Split boundaries remain strict.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        stream = array("q", self.iter_token_ids(split))
        lane_length = len(stream) // batch_size
        if lane_length < 2:
            return
        first = True
        for offset in range(0, lane_length - 1, sequence_length):
            length = min(sequence_length, lane_length - 1 - offset)
            if drop_last and length < sequence_length:
                break
            inputs = tuple(
                tuple(
                    stream[
                        lane * lane_length + offset :
                        lane * lane_length + offset + length
                    ]
                )
                for lane in range(batch_size)
            )
            targets = tuple(
                tuple(
                    stream[
                        lane * lane_length + offset + 1 :
                        lane * lane_length + offset + length + 1
                    ]
                )
                for lane in range(batch_size)
            )
            yield LMTokenBatch(
                input_ids=_maybe_tensor(inputs, as_tensors, device),
                target_ids=_maybe_tensor(targets, as_tensors, device),
                batch_size=batch_size,
                sequence_length=length,
                reset_state=first,
            )
            first = False


def build_wikitext2_corpus(
    paths: WikiText2Paths,
    *,
    lowercase: bool = False,
    min_frequency: int = 1,
    max_vocab_size: Optional[int] = None,
) -> WikiText2Corpus:
    """Build a deterministic corpus; only ``paths.train`` contributes vocabulary."""

    tokenizer = RegexWordPunctTokenizer(lowercase=lowercase)
    vocabulary = Vocabulary.build(
        tokenizer.iter_file_tokens(paths.train),
        min_frequency=min_frequency,
        max_size=max_vocab_size,
    )
    return WikiText2Corpus(paths=paths, tokenizer=tokenizer, vocabulary=vocabulary)


def load_wikitext2(
    cache_dir: PathLike,
    *,
    download: bool = False,
    force_download: bool = False,
    source: WikiText2Source = WIKITEXT2_RAW_SOURCE,
    timeout: float = 60.0,
    lowercase: bool = False,
    min_frequency: int = 1,
    max_vocab_size: Optional[int] = None,
) -> WikiText2Corpus:
    """Prepare verified WikiText-2 raw data and build its train-only vocabulary."""

    paths = prepare_wikitext2(
        cache_dir,
        download=download,
        force_download=force_download,
        source=source,
        timeout=timeout,
    )
    return build_wikitext2_corpus(
        paths,
        lowercase=lowercase,
        min_frequency=min_frequency,
        max_vocab_size=max_vocab_size,
    )


__all__ = [
    "LMTokenBatch",
    "LMTokenChunk",
    "RegexWordPunctTokenizer",
    "SPECIAL_TOKENS",
    "SPLITS",
    "Vocabulary",
    "WIKITEXT2_RAW_SOURCE",
    "WikiText2Corpus",
    "WikiText2Paths",
    "WikiText2Source",
    "WikiTextDownloadError",
    "WikiTextError",
    "WikiTextIntegrityError",
    "WikiTextNotFoundError",
    "build_wikitext2_corpus",
    "download_wikitext2",
    "file_sha256",
    "load_wikitext2",
    "prepare_wikitext2",
    "verify_wikitext2_archive",
]
