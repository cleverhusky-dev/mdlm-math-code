"""Offline checks for frozen input integrity."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import manage_inputs
finally:
    sys.path.pop(0)


def asset(relative: str, content: bytes, kind: str = "tokenizer") -> dict:
    return {
        "path": relative,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "kind": kind,
        "url": "https://huggingface.co/fixture/resolve/fixed/input.bin",
    }


class InputIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.content = b"verified input fixture"
        self.entry = asset("inputs/fixture.bin", self.content)
        self.path = self.root / self.entry["path"]

    def write_existing(self, content: bytes):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(content)

    def test_existing_correct_input_is_not_downloaded(self):
        self.write_existing(self.content)
        with patch.object(manage_inputs, "urlopen") as network:
            manage_inputs.download_missing(self.root, self.entry)
        network.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.content)

    def test_existing_size_or_digest_mismatch_is_preserved(self):
        for invalid in (b"short", b"x" * len(self.content)):
            with self.subTest(size=len(invalid)):
                self.write_existing(invalid)
                with patch.object(manage_inputs, "urlopen") as network:
                    with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                        manage_inputs.download_missing(self.root, self.entry)
                network.assert_not_called()
                self.assertEqual(self.path.read_bytes(), invalid)
                self.assertFalse(list(self.root.rglob("*.part")))

    def test_bad_download_is_not_published_and_temporary_is_removed(self):
        for invalid in (b"short", b"x" * len(self.content), self.content + b"extra"):
            with self.subTest(size=len(invalid)):
                with patch.object(manage_inputs, "urlopen", return_value=io.BytesIO(invalid)):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(ValueError):
                            manage_inputs.download_missing(self.root, self.entry)
                self.assertFalse(self.path.exists())
                self.assertFalse(list(self.root.rglob("*.part")))

    def test_valid_download_is_published_after_verification(self):
        with patch.object(manage_inputs, "urlopen", return_value=io.BytesIO(self.content)) as network:
            with contextlib.redirect_stdout(io.StringIO()):
                manage_inputs.download_missing(self.root, self.entry)
        network.assert_called_once()
        manage_inputs.verify_file(self.path, self.entry)
        self.assertEqual(self.path.read_bytes(), self.content)
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_network_failure_leaves_no_destination_or_partial_file(self):
        with patch.object(manage_inputs, "urlopen", side_effect=URLError("offline fixture")):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(URLError):
                    manage_inputs.download_missing(self.root, self.entry)
        self.assertFalse(self.path.exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_relative_and_absolute_path_escape_are_rejected(self):
        paths = ("../outside.bin", "nested/../../outside.bin",
                 str(self.root.parent / "outside.bin"))
        for relative in paths:
            with self.subTest(path=relative):
                with self.assertRaisesRegex(ValueError, "escapes project"):
                    manage_inputs.asset_path(self.root, relative)
                malicious = {**self.entry, "path": relative}
                (self.root / "inputs.lock.json").write_text(
                    json.dumps({"schema_version": 1, "files": [malicious]}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "escapes project"):
                    manage_inputs.load_assets(self.root)



if __name__ == "__main__":
    unittest.main()
