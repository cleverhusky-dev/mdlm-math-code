"""Shared setup checks without installing packages or using a GPU/network."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_environment", ROOT / "scripts/setup_environment.py")
setup_environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_environment)


class SharedSetupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.commands = []
        self.version = [3, 12]

    def make_venv(self, *, inherit=False):
        env = self.root / ".venv"
        (env / "bin").mkdir(parents=True)
        (env / "bin/python").write_text("fixture", encoding="utf-8")
        (env / "pyvenv.cfg").write_text(
            f"include-system-site-packages = {'true' if inherit else 'false'}\n", encoding="utf-8")

    def fake_run(self, command, **kwargs):
        self.commands.append(command)
        output = ""
        if command[1:4] == ["-I", "-m", "venv"]:
            self.make_venv()
        if len(command) > 3 and "'base_prefix':sys.base_prefix" in command[3]:
            output = json.dumps({"version": self.version, "prefix": str(self.root / ".venv"),
                                 "base_prefix": str(self.root / "platform-python")})
        if "freeze" in command:
            output = "torch==2.6.0+cu124\n"
        return subprocess.CompletedProcess(command, 0, output)

    def run_setup(self, **kwargs):
        with patch.object(setup_environment.sys, "platform", "linux"), \
             patch.object(setup_environment.sys, "version_info", (3, 12, 0)), \
             patch.object(setup_environment.subprocess, "run", side_effect=self.fake_run), \
             patch.object(os, "environ", os.environ.copy()), contextlib.redirect_stdout(io.StringIO()):
            setup_environment.setup(self.root, **kwargs)

    def test_no_gpu_setup_uses_cuda_wheel_and_verifies_inputs_first(self):
        self.run_setup()
        self.assertTrue(self.commands[0][2].endswith("manage_inputs.py"))
        self.assertNotIn("--download", self.commands[0])
        installs = [c for c in self.commands if "install" in c]
        self.assertEqual(len(installs), 2)
        self.assertIn("torch==2.6.0+cu124", installs[0])
        self.assertIn("https://download.pytorch.org/whl/cu124", installs[0])
        self.assertTrue(all("--isolated" in c and "--no-cache-dir" in c for c in installs))
        self.assertTrue(any("check" in c for c in self.commands))
        probe = next(c[3] for c in self.commands if len(c) > 3 and "preflight_executor" in c[3])
        self.assertIn("validate_environment()", probe)
        self.assertNotIn("cuda.is_available", probe)
        self.assertEqual((self.root / ".venv/installed-requirements.txt").read_text(), "torch==2.6.0+cu124\n")

    def test_download_is_explicit_and_existing_environment_is_reused(self):
        self.make_venv()
        self.run_setup(download=True)
        inputs = next(c for c in self.commands if len(c) > 2 and c[2].endswith("manage_inputs.py"))
        self.assertIn("--download", inputs)
        self.assertFalse(any(c[1:4] == ["-I", "-m", "venv"] for c in self.commands))

    def test_existing_system_site_environment_is_rejected_before_install(self):
        self.make_venv(inherit=True)
        with self.assertRaisesRegex(RuntimeError, "include-system-site-packages"):
            self.run_setup()
        self.assertFalse(self.commands)

    def test_existing_wrong_python_version_is_rejected_before_install(self):
        self.make_venv()
        self.version = [3, 10]
        with self.assertRaisesRegex(RuntimeError, "Python 3.12"):
            self.run_setup()
        self.assertFalse(any("install" in c for c in self.commands))

    def test_bad_inputs_stop_before_creating_environment(self):
        def fail_inputs(command, **kwargs):
            self.commands.append(command)
            raise subprocess.CalledProcessError(1, command)
        with patch.object(setup_environment.sys, "platform", "linux"), \
             patch.object(setup_environment.sys, "version_info", (3, 12, 0)), \
             patch.object(setup_environment.subprocess, "run", side_effect=fail_inputs), \
             patch.object(os, "environ", os.environ.copy()):
            with self.assertRaises(subprocess.CalledProcessError):
                setup_environment.setup(self.root)
        self.assertEqual(len(self.commands), 1)
        self.assertFalse((self.root / ".venv").exists())

    def test_shell_help_and_unknown_argument_do_not_install(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "")
        if not bash.is_file():
            self.skipTest("Bash unavailable")
        for arguments, code in ((["--help"], 0), (["--not-an-option"], 2)):
            result = subprocess.run([str(bash), str(ROOT / "setup.sh"), *arguments],
                                    text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, code, result.stdout + result.stderr)
            if code == 0:
                self.assertIn("PYTHON_BIN", result.stdout)
                self.assertIn("GPU is not required", result.stdout)


if __name__ == "__main__":
    unittest.main()
