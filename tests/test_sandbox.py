import platform

import pytest

from charlie_alpha.sandbox import (
    evaluate_function_candidate,
    evaluate_standalone_candidate,
    sandbox_self_test,
)


@pytest.mark.skipif(platform.system() != "Darwin", reason="sandbox-exec is macOS-only")
def test_sandbox_blocks_network_and_external_writes() -> None:
    result = sandbox_self_test()
    assert result["passed"], result


@pytest.mark.skipif(platform.system() != "Darwin", reason="sandbox-exec is macOS-only")
def test_function_candidate_executes_in_sandbox() -> None:
    result = evaluate_function_candidate(
        candidate_code="def add(a, b):\n    return a + b\n",
        prompt="def add(a, b):\n    pass\n",
        canonical_solution="def add(a, b):\n    return a + b\n",
        entry_point="add",
        inputs=[(1, 2), (-1, 3)],
        atol=0.0,
    )
    assert result["passed"], result
    assert result["sandboxed"]


@pytest.mark.skipif(platform.system() != "Darwin", reason="sandbox-exec is macOS-only")
def test_cpp_candidate_compiles_and_executes_in_sandbox() -> None:
    result = evaluate_standalone_candidate(
        candidate_code=(
            "#include <iostream>\n"
            "int main(){long long a,b; std::cin>>a>>b; std::cout<<a+b<<'\\n';}\n"
        ),
        language="cpp",
        tests=[{"input": "2 3\n", "output": "5\n"}],
    )
    assert result["passed"], result
    assert result["sandboxed"]
