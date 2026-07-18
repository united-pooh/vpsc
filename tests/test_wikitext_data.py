import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from vpsc.world_model.wikitext import (
    WIKITEXT2_RAW_SOURCE,
    WikiText2Source,
    WikiTextIntegrityError,
    WikiTextNotFoundError,
    file_sha256,
    load_wikitext2,
    prepare_wikitext2,
)


class WikiTextDataTests(unittest.TestCase):
    def _tiny_source(self, root: Path) -> WikiText2Source:
        archive = root / "tiny-source.zip"
        contents = {
            "tiny/wiki.train.raw": "alpha beta.\ngamma!\n",
            "tiny/wiki.valid.raw": "validonly alpha?\n",
            "tiny/wiki.test.raw": "testonly beta.\n",
        }
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for member, text in contents.items():
                handle.writestr(member, text.encode("utf-8"))
        return WikiText2Source(
            name="tiny-wikitext-2-raw-v1",
            version="fixture-1",
            url=archive.as_uri(),
            sha256=file_sha256(archive),
            archive_filename="tiny-wikitext-2-raw-v1.zip",
            archive_bytes=archive.stat().st_size,
            split_members=(
                ("train", "tiny/wiki.train.raw"),
                ("valid", "tiny/wiki.valid.raw"),
                ("test", "tiny/wiki.test.raw"),
            ),
            homepage="file fixture",
            license="test only",
        )

    def test_pinned_canonical_source_metadata(self) -> None:
        self.assertEqual(
            WIKITEXT2_RAW_SOURCE.sha256,
            "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
        )
        self.assertEqual(WIKITEXT2_RAW_SOURCE.archive_bytes, 4_721_645)
        self.assertTrue(WIKITEXT2_RAW_SOURCE.url.startswith("https://"))

    def test_offline_fixture_train_only_vocab_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._tiny_source(root)
            corpus = load_wikitext2(root / "cache", download=True, source=source)

            self.assertIn("alpha", corpus.vocabulary)
            self.assertNotIn("validonly", corpus.vocabulary)
            self.assertNotIn("testonly", corpus.vocabulary)
            self.assertEqual(
                corpus.encode_text("validonly"), (corpus.vocabulary.unk_id,)
            )
            self.assertEqual(
                tuple(corpus.iter_tokens("train")),
                ("<bos>", "alpha", "beta", ".", "<eos>", "gamma", "!", "<eos>"),
            )

            with corpus.paths.manifest.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["source"], source.metadata())
            self.assertEqual(manifest["archive"]["sha256"], source.sha256)

            cached = load_wikitext2(root / "cache", download=False, source=source)
            self.assertEqual(
                cached.vocabulary.fingerprint, corpus.vocabulary.fingerprint
            )

    def test_chunks_and_batches_are_contiguous_with_explicit_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._tiny_source(root)
            corpus = load_wikitext2(root / "cache", download=True, source=source)

            chunks = list(
                corpus.iter_chunks("train", sequence_length=3, drop_last=False)
            )
            self.assertEqual([chunk.length for chunk in chunks], [3, 3, 1])
            self.assertTrue(chunks[0].reset_state)
            self.assertTrue(all(not chunk.reset_state for chunk in chunks[1:]))
            for previous, current in zip(chunks, chunks[1:]):
                self.assertEqual(previous.target_ids[-1], current.input_ids[0])

            batches = list(
                corpus.iter_batches(
                    "train", batch_size=2, sequence_length=2, drop_last=False
                )
            )
            self.assertEqual([batch.sequence_length for batch in batches], [2, 1])
            self.assertTrue(batches[0].reset_state)
            self.assertFalse(batches[1].reset_state)
            for lane in range(2):
                self.assertEqual(
                    batches[0].target_ids[lane][-1], batches[1].input_ids[lane][0]
                )

            valid_ids = tuple(corpus.iter_token_ids("valid"))
            self.assertEqual(valid_ids[0], corpus.vocabulary.bos_id)
            self.assertIn(corpus.vocabulary.unk_id, valid_ids)

    def test_missing_cache_never_downloads_without_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._tiny_source(root)
            with self.assertRaises(WikiTextNotFoundError):
                prepare_wikitext2(root / "empty-cache", source=source)
            self.assertFalse((root / "empty-cache").exists())

    def test_corrupt_cached_archive_is_rejected_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._tiny_source(root)
            paths = prepare_wikitext2(root / "cache", download=True, source=source)
            with paths.archive.open("ab") as handle:
                handle.write(b"corruption")

            with self.assertRaises(WikiTextIntegrityError):
                prepare_wikitext2(root / "cache", download=False, source=source)


if __name__ == "__main__":
    unittest.main()
