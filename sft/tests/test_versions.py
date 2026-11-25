"""Verify source identities in Git and portable source directories."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_version import capture_version, git


@unittest.skipUnless(shutil.which("git"), "Optional Git checkout tests require Git")
class ExperimentVersionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        git(self.root, "init", "-b", "main")
        git(self.root, "config", "user.name", "Fixture")
        git(self.root, "config", "user.email", "fixture@example.invalid")
        git(self.root, "config", "core.autocrlf", "false")
        self.write(".gitattributes", "* text=auto\n*.py text eol=lf\n*.sh text eol=lf\n*.json text eol=lf\n")
        self.write(".gitignore", "*.bundle\n*.bundle.sha256\n.bundle-*/\ninputs/*.ckpt\n.env\nruns/\n")
        self.write("experiments/experiment0/config.json", '{"seed": 1}\n')
        self.write("experiments/experiment0/run.sh", "#!/usr/bin/env bash\nexit 0\n")
        self.write("model.py", "value = 1\n")
        self.write("inputs/mdlm.ckpt", "large input fixture not tracked")
        self.write(".env", "not a real credential")
        git(self.root, "add", ".")
        git(self.root, "commit", "-m", "Fixture experiment")
        git(self.root, "tag", "-a", "experiment0-v1", "-m", "Fixture")

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode())

    def test_clean_version_records_tracked_sources_only(self):
        record = capture_version(self.root)
        self.assertFalse(record["dirty"])
        self.assertEqual(record["tags"], ["experiment0-v1"])
        self.assertEqual(record["tracked_files_sha256"]["model.py"], hashlib.sha256(b"value = 1\n").hexdigest())
        self.assertNotIn(".env", record["tracked_files_sha256"])
        self.assertNotIn("inputs/mdlm.ckpt", record["tracked_files_sha256"])

    def test_modified_and_untracked_code_are_rejected(self):
        for name in ("model.py", "extra.py"):
            with self.subTest(name=name):
                self.write(name, "value = 2\n")
                with self.assertRaisesRegex(RuntimeError, "Uncommitted"):
                    capture_version(self.root)
                if name == "model.py":
                    git(self.root, "restore", name)
                else:
                    (self.root / name).unlink()


    def test_empty_project_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "No project source"):
            capture_version(self.base)


class PortableVersionTests(unittest.TestCase):
    def test_copy_keeps_identity_and_source_changes_are_detected(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            (root / "mdlm_sft").mkdir(parents=True)
            model = root / "mdlm_sft/model.py"
            model.write_text("value = 1\n", encoding="utf-8")
            first = capture_version(root)
            self.assertIsNone(first["commit"])
            self.assertEqual(first["source_mode"], "file_snapshot")
            other = Path(directory) / "renamed"
            shutil.copytree(root, other)
            self.assertEqual(first, capture_version(other))
            model.write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(first, capture_version(root))
            model.write_text("value = 1\n", encoding="utf-8")
            extra = root / "mdlm_sft/extra.py"
            extra.write_text("extra = True\n", encoding="utf-8")
            self.assertNotEqual(first, capture_version(root))
            extra.unlink()
            self.assertEqual(first, capture_version(root))

    def test_results_weights_and_caches_do_not_change_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("value = 1\n", encoding="utf-8")
            first = capture_version(root)
            for name in ("runs/result.json", "inputs/mdlm.ckpt", "data/prepared/manifest.json",
                         "scripts/__pycache__/cached.py", "README.md", ".env"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("excluded fixture", encoding="utf-8")
            self.assertEqual(first, capture_version(root))

    def test_configuration_is_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "experiments/experiment12/config.json"
            config.parent.mkdir(parents=True)
            config.write_text('{"seed": 1}', encoding="utf-8")
            first = capture_version(root)
            self.assertIn("experiments/experiment12/config.json", first["tracked_files_sha256"])
            config.write_text('{"seed": 2}', encoding="utf-8")
            self.assertNotEqual(first["snapshot_sha256"], capture_version(root)["snapshot_sha256"])


if __name__ == "__main__":
    unittest.main()
