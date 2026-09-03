from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

from .io_utils import atomic_write_text


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))


def _profile(directory: Path) -> str:
    escaped = str(directory.resolve()).replace('"', '\\"')
    return f"""(version 1)
(deny default)
(allow process*)
(allow file-read*)
(allow file-write* (subpath "{escaped}"))
(deny network*)
(allow sysctl-read)
(allow mach-lookup)
"""


def _run(
    command: list[str], directory: Path, *, input_text: str | None = None, timeout: int = 8
) -> subprocess.CompletedProcess[str]:
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": str(directory),
        "HOME": str(directory),
        "PYTHONHASHSEED": "0",
    }
    full_command = ["/usr/bin/sandbox-exec", "-p", _profile(directory), *command]
    process = subprocess.Popen(
        full_command,
        cwd=directory,
        env=environment,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_limits,
        start_new_session=True,
    )
    if input_text is not None and process.stdin is not None:
        process.stdin.write(input_text)
        process.stdin.close()
        process.stdin = None
    started = time.monotonic()
    memory_limit = 1536 * 1024 * 1024
    while process.poll() is None:
        if time.monotonic() - started >= timeout:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise subprocess.TimeoutExpired(full_command, timeout)
        try:
            root = psutil.Process(process.pid)
            resident = root.memory_info().rss + sum(
                child.memory_info().rss for child in root.children(recursive=True)
            )
            if resident > memory_limit:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return subprocess.CompletedProcess(
                    full_command,
                    -signal.SIGKILL,
                    stdout="",
                    stderr="memory limit exceeded",
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    return subprocess.CompletedProcess(
        full_command,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def evaluate_function_candidate(
    *,
    candidate_code: str,
    prompt: str,
    canonical_solution: str,
    entry_point: str,
    inputs: list[Any],
    atol: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="charlie-alpha-eval-") as temporary:
        directory = Path(temporary)
        candidate_source = (
            candidate_code
            if f"def {entry_point}" in candidate_code
            else f"{prompt.rstrip()}\n{candidate_code}"
        )
        trusted_source = f"{prompt.rstrip()}\n{canonical_solution}"
        runner = f"""import copy
import json
import math

candidate_namespace = {{}}
trusted_namespace = {{}}
exec({candidate_source!r}, candidate_namespace)
exec({trusted_source!r}, trusted_namespace)
candidate = candidate_namespace[{entry_point!r}]
trusted = trusted_namespace[{entry_point!r}]
inputs = {inputs!r}
atol = {float(atol)!r}

def equivalent(left, right):
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=atol)
        except (TypeError, ValueError):
            return False
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, set) or isinstance(right, set):
        try:
            return set(left) == set(right)
        except TypeError:
            return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    return left == right

passed = 0
for arguments in inputs:
    expected = trusted(*copy.deepcopy(arguments))
    actual = candidate(*copy.deepcopy(arguments))
    if not equivalent(actual, expected):
        print(json.dumps({{"passed": False, "passed_tests": passed, "reason": "wrong answer"}}))
        raise SystemExit(0)
    passed += 1
print(json.dumps({{"passed": True, "passed_tests": passed}}))
"""
        runner_path = directory / "runner.py"
        atomic_write_text(runner_path, runner)
        try:
            result = _run([sys.executable, str(runner_path)], directory, timeout=10)
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": "timeout", "sandboxed": True}
        if result.returncode != 0:
            return {
                "passed": False,
                "reason": "runtime error",
                "stderr": result.stderr[-1000:],
                "sandboxed": True,
            }
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            payload = {"passed": False, "reason": "invalid runner output"}
        payload["sandboxed"] = True
        return payload


def evaluate_standalone_candidate(
    *, candidate_code: str, language: str, tests: list[dict[str, str]]
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="charlie-alpha-eval-") as temporary:
        directory = Path(temporary)
        if language == "python":
            source_path = directory / "solution.py"
            atomic_write_text(source_path, candidate_code)
            command = [sys.executable, str(source_path)]
        elif language == "cpp":
            source_path = directory / "solution.cpp"
            binary_path = directory / "solution"
            atomic_write_text(source_path, candidate_code)
            command_line_tools_clang = Path("/Library/Developer/CommandLineTools/usr/bin/clang++")
            compiler = (
                command_line_tools_clang
                if command_line_tools_clang.exists()
                else Path("/usr/bin/clang++")
            )
            sdk_path = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
            sysroot_arguments = ["-isysroot", str(sdk_path)] if sdk_path.exists() else []
            try:
                compilation = _run(
                    [
                        str(compiler),
                        *sysroot_arguments,
                        "-std=c++17",
                        "-O2",
                        str(source_path),
                        "-o",
                        str(binary_path),
                    ],
                    directory,
                    timeout=15,
                )
            except subprocess.TimeoutExpired:
                return {"passed": False, "reason": "compile timeout", "sandboxed": True}
            if compilation.returncode != 0:
                return {
                    "passed": False,
                    "reason": "compile error",
                    "stderr": compilation.stderr[-1000:],
                    "sandboxed": True,
                }
            command = [str(binary_path)]
        else:
            return {"passed": False, "reason": "unsupported language", "sandboxed": True}

        passed = 0
        for test in tests[:10]:
            try:
                result = _run(command, directory, input_text=test["input"], timeout=5)
            except subprocess.TimeoutExpired:
                return {"passed": False, "reason": "timeout", "sandboxed": True}
            expected = "\n".join(line.rstrip() for line in test["output"].strip().splitlines())
            actual = "\n".join(line.rstrip() for line in result.stdout.strip().splitlines())
            if result.returncode != 0 or actual != expected:
                return {
                    "passed": False,
                    "passed_tests": passed,
                    "reason": "runtime error" if result.returncode else "wrong answer",
                    "sandboxed": True,
                }
            passed += 1
        return {"passed": True, "passed_tests": passed, "sandboxed": True}


def sandbox_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="charlie-alpha-sandbox-test-") as temporary:
        directory = Path(temporary)
        script = """import json
import socket

network_blocked = False
write_blocked = False
try:
    socket.create_connection(("1.1.1.1", 53), timeout=1)
except OSError:
    network_blocked = True
try:
    with open("/tmp/charlie-alpha-sandbox-escape", "w") as handle:
        handle.write("escape")
except OSError:
    write_blocked = True
print(json.dumps({"network_blocked": network_blocked, "write_blocked": write_blocked}))
"""
        script_path = directory / "self_test.py"
        atomic_write_text(script_path, script)
        try:
            result = _run([sys.executable, str(script_path)], directory, timeout=5)
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": "timeout"}
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return {"passed": False, "reason": result.stderr[-1000:]}
        payload["passed"] = bool(
            result.returncode == 0
            and payload.get("network_blocked")
            and payload.get("write_blocked")
        )
        return payload
