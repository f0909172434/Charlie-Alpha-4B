from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from .io_utils import atomic_write_text, write_json


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: int = 20
    memory_bytes: int = 2 * 1024**3
    max_write_bytes: int = 32 * 1024**2
    max_output_bytes: int = 64 * 1024
    cpu_seconds: int = 20


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    memory_exceeded: bool
    write_exceeded: bool
    output_exceeded: bool
    isolated: bool
    network_allowed: bool
    elapsed_seconds: float
    written_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _escaped(path: Path) -> str:
    return str(path.resolve()).replace('"', '\\"')


def _escaped_literal(path: Path) -> str:
    return str(path.absolute()).replace('"', '\\"')


def _runtime_roots(executable: Path) -> list[Path]:
    roots = {
        executable.parent,
        executable.parent.parent,
        executable.resolve().parent,
        Path(os.path.realpath(executable)).parent,
    }
    prefix = Path(os.path.realpath(executable)).parent.parent
    roots.add(prefix)
    if executable.name == "R" and executable.parent.name == "exec":
        roots.add(executable.parents[2])
        roots.add(executable.parents[4])
    return sorted(roots)


def _profile(
    directory: Path,
    executable: Path,
    extra_read_roots: list[Path],
    additional_executables: list[Path],
) -> str:
    read_roots = [directory, *_runtime_roots(executable), *extra_read_roots]
    read_rules = "\n".join(
        f'(allow file-read-data (subpath "{_escaped(root)}"))' for root in read_roots
    )
    executable_rule = f'(allow process-exec (literal "{_escaped_literal(executable)}"))'
    real_executable = executable.resolve()
    if real_executable != executable:
        executable_rule += f'\n(allow process-exec (literal "{_escaped(real_executable)}"))'
    for approved in additional_executables:
        executable_rule += f'\n(allow process-exec (literal "{_escaped_literal(approved)}"))'
        resolved = approved.resolve()
        if resolved != approved:
            executable_rule += f'\n(allow process-exec (literal "{_escaped(resolved)}"))'
    return f"""(version 1)
(deny default)
(allow file-read-data)
(allow file-read-metadata)
(deny file-read-data (subpath "/Users"))
(deny file-read-data (subpath "/Volumes"))
(deny file-read-data (subpath "/Network"))
{read_rules}
(allow file-read* (subpath "/System/Library"))
(allow file-read* (subpath "/usr/lib"))
(allow file-read* (literal "/private/etc/localtime"))
(allow file-read* (literal "/dev/null"))
(allow file-read* (literal "/dev/urandom"))
(allow file-write-data (literal "/dev/null"))
(allow file-write* (subpath "{_escaped(directory)}"))
{executable_rule}
(allow process-fork)
(allow process-info*)
(allow sysctl-read)
(allow mach-lookup)
(deny network*)
"""


def _set_limits(limits: SandboxLimits) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_write_bytes, limits.max_write_bytes))
    if hasattr(resource, "RLIMIT_AS"):
        with suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))


def _directory_size(directory: Path, *, ignored: set[Path]) -> int:
    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            path = (Path(root) / name).resolve()
            if path in ignored:
                continue
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def run_isolated(
    *,
    executable: Path,
    arguments: list[str],
    directory: Path,
    limits: SandboxLimits,
    extra_read_roots: list[Path] | None = None,
    additional_executables: list[Path] | None = None,
) -> SandboxResult:
    executable = executable.expanduser().absolute()
    directory = directory.expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Sandbox runtime does not exist: {executable}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Sandbox directory does not exist: {directory}")
    stdout_path = directory / ".stdout"
    stderr_path = directory / ".stderr"
    ignored = {stdout_path.resolve(), stderr_path.resolve()}
    baseline = _directory_size(directory, ignored=ignored)
    environment = {
        "PATH": f"{executable.parent}:/usr/bin:/bin",
        "TMPDIR": str(directory),
        "HOME": str(directory),
        "PYTHONHASHSEED": "0",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "R_USER": str(directory),
        "R_ENVIRON_USER": str(directory / ".Renviron-disabled"),
        "R_PROFILE_USER": str(directory / ".Rprofile-disabled"),
        "KMP_INIT_AT_FORK": "FALSE",
        "KMP_USE_SHM": "0",
    }
    if executable.name == "R" and executable.parent.name == "exec":
        environment["R_HOME"] = str(executable.parents[2])
    command = [
        "/usr/bin/sandbox-exec",
        "-p",
        _profile(
            directory,
            executable,
            extra_read_roots or [],
            additional_executables or [],
        ),
        str(executable),
        *arguments,
    ]
    started = time.monotonic()
    timed_out = False
    memory_exceeded = False
    write_exceeded = False
    output_exceeded = False
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            preexec_fn=lambda: _set_limits(limits),
            start_new_session=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= limits.timeout_seconds:
                timed_out = True
            try:
                root = psutil.Process(process.pid)
                resident = root.memory_info().rss + sum(
                    child.memory_info().rss for child in root.children(recursive=True)
                )
                memory_exceeded = resident > limits.memory_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            written = max(0, _directory_size(directory, ignored=ignored) - baseline)
            write_exceeded = written > limits.max_write_bytes
            output_size = sum(
                path.stat().st_size if path.exists() else 0 for path in (stdout_path, stderr_path)
            )
            output_exceeded = output_size > limits.max_output_bytes
            if timed_out or memory_exceeded or write_exceeded or output_exceeded:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                break
            time.sleep(0.02)
        returncode = int(process.wait())
    stdout_bytes = stdout_path.read_bytes()[: limits.max_output_bytes]
    remaining = max(0, limits.max_output_bytes - len(stdout_bytes))
    stderr_bytes = stderr_path.read_bytes()[:remaining]
    written_bytes = max(0, _directory_size(directory, ignored=ignored) - baseline)
    return SandboxResult(
        returncode=returncode,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        timed_out=timed_out,
        memory_exceeded=memory_exceeded,
        write_exceeded=write_exceeded,
        output_exceeded=output_exceeded,
        isolated=True,
        network_allowed=False,
        elapsed_seconds=time.monotonic() - started,
        written_bytes=written_bytes,
    )


class StatsToolSession:
    def __init__(
        self,
        *,
        python_executable: Path,
        r_executable: Path | None,
        limits: SandboxLimits,
        max_calls: int = 4,
    ) -> None:
        self.python_executable = python_executable.expanduser().absolute()
        self.r_executable = r_executable.expanduser().absolute() if r_executable else None
        self.limits = limits
        self.max_calls = max_calls
        self.calls = 0
        self._temporary = tempfile.TemporaryDirectory(prefix="charlie-alpha-stats-")
        self.directory = Path(self._temporary.name).resolve()
        self.data_files: list[Path] = []

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> StatsToolSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def add_files(
        self,
        paths: list[Path],
        *,
        allowed_extensions: set[str],
        max_files: int,
        max_file_bytes: int,
        max_total_bytes: int,
    ) -> list[Path]:
        if len(paths) > max_files:
            raise ValueError(f"At most {max_files} input files are allowed")
        total = 0
        copied: list[Path] = []
        for index, original in enumerate(paths):
            declared = original.expanduser()
            extension = declared.suffix.lower()
            if extension not in allowed_extensions:
                raise ValueError(f"Unsupported input file: {declared.name}")
            source = declared.resolve(strict=True)
            if not source.is_file():
                raise ValueError(f"Unsupported input file: {declared.name}")
            size = source.stat().st_size
            if size > max_file_bytes:
                raise ValueError(f"Input file exceeds the per-file limit: {source.name}")
            total += size
            if total > max_total_bytes:
                raise ValueError("Input files exceed the combined size limit")
            destination = self.directory / f"input-{index}{extension}"
            shutil.copyfile(source, destination)
            destination.chmod(0o400)
            copied.append(destination)
        self.data_files = copied
        return copied

    def _call(
        self,
        executable: Path,
        arguments: list[str],
        *,
        additional_executables: list[Path] | None = None,
    ) -> SandboxResult:
        if self.calls >= self.max_calls:
            raise RuntimeError(f"Statistical tool call limit reached ({self.max_calls})")
        self.calls += 1
        return run_isolated(
            executable=executable,
            arguments=arguments,
            directory=self.directory,
            limits=self.limits,
            additional_executables=additional_executables,
        )

    def run_python(self, script: str, request: dict[str, Any]) -> SandboxResult:
        script_path = self.directory / f"python-tool-{self.calls + 1}.py"
        request_path = self.directory / f"request-{self.calls + 1}.json"
        atomic_write_text(script_path, script)
        write_json(request_path, request)
        return self._call(
            self.python_executable,
            ["-I", "-B", str(script_path), str(request_path)],
        )

    def run_r(self, script: str, request: dict[str, Any]) -> SandboxResult:
        if self.r_executable is None:
            raise RuntimeError("The locked R runtime is not installed")
        script_path = self.directory / f"r-tool-{self.calls + 1}.R"
        request_path = self.directory / f"request-{self.calls + 1}.json"
        atomic_write_text(script_path, script)
        write_json(request_path, request)
        return self._call(
            self.r_executable,
            [
                "--no-echo",
                "--no-restore",
                f"--file={script_path}",
                "--args",
                str(request_path),
            ],
            additional_executables=[
                Path("/bin/sh"),
                Path("/bin/bash"),
                Path("/usr/bin/which"),
                Path("/usr/bin/uname"),
                Path("/usr/bin/sed"),
                Path("/bin/rm"),
            ],
        )


def sandbox_self_test(
    *,
    python_executable: Path,
    r_executable: Path | None = None,
) -> dict[str, Any]:
    limits = SandboxLimits(timeout_seconds=10, cpu_seconds=8)
    forbidden = Path(__file__).resolve()
    with StatsToolSession(
        python_executable=python_executable,
        r_executable=r_executable,
        limits=limits,
    ) as session:
        python_script = f"""import json
import pathlib
import socket
import subprocess

def blocked(call):
    try:
        call()
        return False
    except BaseException:
        return True

result = {{
    "network_blocked": blocked(lambda: socket.create_connection(("1.1.1.1", 53), 1)),
    "read_blocked": blocked(lambda: pathlib.Path({str(forbidden)!r}).read_text()),
    "write_blocked": blocked(lambda: pathlib.Path("/tmp/charlie-alpha-escape").write_text("x")),
    "subprocess_blocked": blocked(lambda: subprocess.run(["/usr/bin/id"], check=True)),
}}
print(json.dumps(result))
"""
        python_result = session.run_python(python_script, {})
        try:
            python_checks = json.loads(python_result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            python_checks = {}
        python_passed = python_result.returncode == 0 and all(
            python_checks.get(key)
            for key in (
                "network_blocked",
                "read_blocked",
                "write_blocked",
                "subprocess_blocked",
            )
        )
        result: dict[str, Any] = {
            "python": {
                "passed": python_passed,
                "checks": python_checks,
                "sandbox": python_result.to_dict(),
            }
        }
        if r_executable is not None:
            r_script = f"""args <- commandArgs(trailingOnly=TRUE)
network_blocked <- inherits(try(url("https://example.com", open="r"), silent=TRUE), "try-error")
read_blocked <- inherits(try(readLines({json.dumps(str(forbidden))}), silent=TRUE), "try-error")
write_blocked <- inherits(try(
  writeLines("x", "/tmp/charlie-alpha-r-escape"), silent=TRUE), "try-error")
subprocess_status <- suppressWarnings(try(system2("/usr/bin/id"), silent=TRUE))
subprocess_blocked <- inherits(subprocess_status, "try-error") ||
  !identical(as.integer(subprocess_status), 0L)
cat(sprintf('{{"network_blocked":%s,"read_blocked":%s,"write_blocked":%s,"subprocess_blocked":%s}}\n',
  tolower(network_blocked), tolower(read_blocked),
  tolower(write_blocked), tolower(subprocess_blocked)))
"""
            r_result = session.run_r(r_script, {})
            try:
                r_checks = json.loads(r_result.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                r_checks = {}
            r_passed = r_result.returncode == 0 and all(
                r_checks.get(key)
                for key in (
                    "network_blocked",
                    "read_blocked",
                    "write_blocked",
                    "subprocess_blocked",
                )
            )
            result["r"] = {
                "passed": r_passed,
                "checks": r_checks,
                "sandbox": r_result.to_dict(),
            }
        result["passed"] = all(
            value.get("passed", False) for key, value in result.items() if key in {"python", "r"}
        )
        return result
