from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from charlie_alpha import stats_opened_source_residual_repair as h21
from charlie_alpha.config import ProjectConfig, load_config
from charlie_alpha.io_utils import canonical_hash, write_json
from charlie_alpha.stats_guarded_external import _DECODING, _messages

_ROOT = Path(__file__).resolve().parents[1]
_REAL_SNAPSHOTS = (
    _ROOT / "data" / "evolve" / "guarded-external-v1" / "source-screen" / "snapshots"
)


def _config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        path=tmp_path / "configs" / "pipeline.evolve.yaml",
        root=tmp_path,
        values={
            "paths": {
                "artifact_dir": "artifacts/evolve",
                "evolution_dir": "data/evolve",
            }
        },
        sources={},
    )


def _copy_snapshots(tmp_path: Path) -> None:
    target = (
        tmp_path
        / "data"
        / "evolve"
        / "guarded-external-v1"
        / "source-screen"
        / "snapshots"
    )
    target.mkdir(parents=True)
    for source_id in h21._SOURCES:
        shutil.copy2(_REAL_SNAPSHOTS / f"{source_id}.bioc.json", target)


def _master() -> dict:
    runtime = {
        "control": {
            "adapter_path": "parent-adapter",
            "adapter_sha256": "parent-sha256",
            "adapter_config_sha256": "parent-config-sha256",
        },
        "repair": {
            "adapter_path": "h20-adapter",
            "adapter_sha256": h21._H20_ADAPTER_SHA256,
            "adapter_config_sha256": "h20-config-sha256",
        },
        "decoding": dict(_DECODING),
        "prompt_messages_sha256": canonical_hash(_messages({"question": "<SOURCE_QUESTION>"})),
    }
    runtime["fingerprint"] = canonical_hash(runtime)
    master = {
        "schema_version": 1,
        "runtime": runtime,
        "source_protocol": {
            "opened_source_exclusions": {
                source_id: {
                    "snapshot_sha256": source["snapshot_sha256"],
                    "status": "opened-during-precontract-feasibility-screen",
                }
                for source_id, source in h21._SOURCES.items()
            }
        },
        "source_selected": False,
        "model_output_opened": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    master["fingerprint"] = canonical_hash(master)
    return master


def _screen(master: dict) -> dict:
    screen = {
        "schema_version": 1,
        "complete": True,
        "master_contract_fingerprint": master["fingerprint"],
        "source_screen_not_blind": True,
        "screened_sources": {
            source_id: {
                "snapshot_sha256": source["snapshot_sha256"],
                "opened_before_master_contract": True,
                "qualified": False,
            }
            for source_id, source in h21._SOURCES.items()
        },
        "status": "SOURCE_SCREEN_UNQUALIFIED",
        "evaluation_authorized": False,
        "model_output_opened": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    screen["result_fingerprint"] = canonical_hash(screen)
    return screen


def _patch_seals(monkeypatch: pytest.MonkeyPatch, master: dict, screen: dict) -> None:
    monkeypatch.setattr(
        h21,
        "prepare_guarded_external_master_contract",
        lambda _config: deepcopy(master),
    )
    monkeypatch.setattr(
        h21,
        "prepare_guarded_external_source_screen",
        lambda _config: deepcopy(screen),
    )


def test_materialization_partitions_all_26_cells_into_24_plus_2() -> None:
    config = load_config(_ROOT / "configs" / "pipeline.evolve.yaml")
    cases, manifest = h21._extract_cases_and_manifest(config)

    assert len(cases) == 24
    assert len({case["source_id"] for case in cases}) == 4
    assert len({case["gold_method_id"] for case in cases}) == 10
    assert "two_proportion" not in {case["gold_method_id"] for case in cases}
    assert sum(case["gold_method_id"] == "mann_whitney" for case in cases) == 3
    assert manifest["eligible_cell_count"] == 26
    assert manifest["included_case_count"] == 24
    assert manifest["ambiguity_exclusion_count"] == 2
    assert len(manifest["eligible_partition"]) == 26
    assert {row["disposition"] for row in manifest["eligible_partition"]} == {
        "included",
        "ambiguity-excluded",
    }
    assert {
        (
            row["source_id"],
            row["table_id"],
            row["row_index"],
            row["column_index"],
            row["gold_method_id"],
        )
        for row in manifest["ambiguity_exclusions"]
    } == {
        ("PMC6639881", "T2", 2, 1, "two_proportion"),
        ("PMC2996580", "T0001", 5, 0, "mann_whitney"),
    }
    expected_fields = {"case_id", "source_id", "question", "gold_method_id"}
    assert all(set(case) == expected_fields for case in cases)
    assert all(case["case_id"].startswith("h21-opened-") for case in cases)


def test_contract_and_data_are_immutable_and_bind_the_reviewed_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _copy_snapshots(tmp_path)
    master = _master()
    screen = _screen(master)
    _patch_seals(monkeypatch, master, screen)

    contract_path = (
        tmp_path / "artifacts/evolve/opened-source-residual-repair-v1/contract.json"
    )
    contract = h21.prepare_opened_source_residual_contract(config)
    contract_bytes = contract_path.read_bytes()
    data = h21.prepare_opened_source_residual_data(config)
    assert h21.prepare_opened_source_residual_contract(config) == contract
    assert contract_path.read_bytes() == contract_bytes
    assert contract["h20_adapter_sha256"] == h21._H20_ADAPTER_SHA256
    assert contract["materialization_contract"]["eligible_source_cell_count"] == 26
    assert contract["materialization_contract"]["included_case_count"] == 24
    assert contract["materialization_contract"]["ambiguity_exclusion_count"] == 2
    assert contract["materialization_contract"]["distinct_method_count"] == 10
    assert contract["opportunity_gates"] == {
        "minimum_residual_invalid_cases": 6,
        "minimum_residual_invalid_sources": 3,
        "minimum_residual_invalid_gold_methods": 3,
    }
    assert data["case_count"] == 24
    assert data["eligible_cell_count"] == 26
    assert data["ambiguity_exclusion_count"] == 2
    assert data["source_count"] == 4
    assert data["distinct_method_count"] == 10
    assert data["training_authorized"] is False
    assert data["fresh_external_evidence"] is False

    drifted = deepcopy(screen)
    drifted["review_drift"] = True
    drifted["result_fingerprint"] = h21._fingerprint_without(
        drifted, "result_fingerprint", "private_details"
    )
    _patch_seals(monkeypatch, master, drifted)
    with pytest.raises(RuntimeError, match="contract is immutable"):
        h21.prepare_opened_source_residual_contract(config)


def test_snapshot_drift_and_question_label_leak_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _copy_snapshots(tmp_path)
    master = _master()
    screen = _screen(master)
    _patch_seals(monkeypatch, master, screen)
    snapshot = (
        tmp_path
        / "data/evolve/guarded-external-v1/source-screen/snapshots/PMC8327789.bioc.json"
    )
    snapshot.write_bytes(snapshot.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="snapshot changed or is missing"):
        h21.prepare_opened_source_residual_data(config)

    shutil.copy2(_REAL_SNAPSHOTS / "PMC8327789.bioc.json", snapshot)
    included = deepcopy(h21._INCLUDED_CELLS)
    key = next(iter(included))
    included[key] = (included[key][0], "Choose the Mann-Whitney test.")
    monkeypatch.setattr(h21, "_INCLUDED_CELLS", included)
    with pytest.raises(RuntimeError, match="forbidden method label or alias"):
        h21._extract_cases_and_manifest(config)


def _candidate(source_id: str, method_id: str, *, valid: bool) -> dict:
    return {
        "source_id": source_id,
        "gold_method_id": method_id,
        "valid_output": valid,
    }


@pytest.mark.parametrize(
    ("residual", "qualified"),
    [
        (
            [
                _candidate("s1", "m1", valid=False),
                _candidate("s1", "m2", valid=False),
                _candidate("s2", "m2", valid=False),
                _candidate("s2", "m3", valid=False),
                _candidate("s3", "m1", valid=False),
                _candidate("s3", "m3", valid=False),
            ],
            True,
        ),
        (
            [
                _candidate("s1", "m1", valid=False),
                _candidate("s1", "m2", valid=False),
                _candidate("s2", "m2", valid=False),
                _candidate("s2", "m3", valid=False),
                _candidate("s3", "m1", valid=False),
            ],
            False,
        ),
        (
            [
                _candidate("s1", "m1", valid=False),
                _candidate("s1", "m2", valid=False),
                _candidate("s1", "m3", valid=False),
                _candidate("s2", "m1", valid=False),
                _candidate("s2", "m2", valid=False),
                _candidate("s2", "m3", valid=False),
            ],
            False,
        ),
        (
            [
                _candidate("s1", "m1", valid=False),
                _candidate("s1", "m2", valid=False),
                _candidate("s2", "m1", valid=False),
                _candidate("s2", "m2", valid=False),
                _candidate("s3", "m1", valid=False),
                _candidate("s3", "m2", valid=False),
            ],
            False,
        ),
    ],
)
def test_opportunity_gate_uses_post_h20_residuals(
    residual: list[dict], qualified: bool
) -> None:
    details = {"candidate": [*residual, _candidate("s4", "m4", valid=True)]}
    result = h21._opportunity_summary(details)
    assert result["qualified"] is qualified
    assert result["status"] == (
        "OPPORTUNITY_QUALIFIED" if qualified else "INCONCLUSIVE_OPPORTUNITY"
    )


def _qualified_residual_ids(cases: list[dict]) -> set[str]:
    residual: list[dict] = [
        case for case in cases if case["source_id"] == "PMC2996580"
    ]
    for source_id in ("PMC6639881", "PMC8327789", "PMC8483143"):
        residual.append(next(case for case in cases if case["source_id"] == source_id))
    assert len(residual) == 6
    assert len({case["source_id"] for case in residual}) >= 3
    assert len({case["gold_method_id"] for case in residual}) >= 3
    return {str(case["case_id"]) for case in residual}


def _install_fake_callers(
    monkeypatch: pytest.MonkeyPatch,
    cases: list[dict],
    *,
    control_invalid_ids: set[str],
    residual_ids: set[str],
) -> type:
    gold_by_id = {str(case["case_id"]): str(case["gold_method_id"]) for case in cases}

    class FakeAgentCaller:
        events: list[str] = []
        records: list[dict] = []

        def __init__(
            self,
            _config: ProjectConfig,
            *,
            adapter_path: str,
            expected_calls: int | None,
        ) -> None:
            self.stage = "parent" if adapter_path == "parent-adapter" else "h20"
            self.expected_calls = expected_calls
            self.calls = 0
            self.loaded = False

        def __call__(
            self,
            case_id: str,
            messages: list[dict[str, str]],
            decoding: dict,
        ) -> str:
            if not self.loaded:
                self.loaded = True
                self.events.append(f"load:{self.stage}")
            self.records.append(
                {
                    "stage": self.stage,
                    "case_id": case_id,
                    "messages": deepcopy(messages),
                    "decoding": deepcopy(decoding),
                }
            )
            if self.stage == "parent":
                answer = (
                    "CONTROL_INVALID"
                    if case_id in control_invalid_ids
                    else json.dumps({"methods": [gold_by_id[case_id]]})
                )
            else:
                assert case_id in control_invalid_ids
                answer = (
                    ""
                    if case_id in residual_ids
                    else json.dumps({"methods": [gold_by_id[case_id]]})
                )
            self.calls += 1
            if self.expected_calls is not None and self.calls == self.expected_calls:
                self.close()
            return answer

        def close(self) -> None:
            if self.loaded:
                self.events.append(f"close:{self.stage}")
                self.loaded = False

    monkeypatch.setattr(h21, "_AgentCaller", FakeAgentCaller)
    return FakeAgentCaller


def test_exact_runner_routes_invalid_only_and_replays_terminal_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _copy_snapshots(tmp_path)
    master = _master()
    screen = _screen(master)
    _patch_seals(monkeypatch, master, screen)
    cases, _ = h21._extract_cases_and_manifest(config)
    residual_ids = _qualified_residual_ids(cases)
    repairable_ids = {
        str(case["case_id"])
        for case in cases
        if str(case["case_id"]) not in residual_ids
    }
    repairable_ids = set(sorted(repairable_ids)[:3])
    control_invalid_ids = residual_ids | repairable_ids
    fake = _install_fake_callers(
        monkeypatch,
        cases,
        control_invalid_ids=control_invalid_ids,
        residual_ids=residual_ids,
    )

    public = h21.run_opened_source_residual_opportunity(config)
    report_path = (
        tmp_path / "artifacts/evolve/opened-source-residual-repair-v1/report.json"
    )
    terminal_bytes = report_path.read_bytes()
    internal = json.loads(terminal_bytes)
    parent_records = [row for row in fake.records if row["stage"] == "parent"]
    h20_records = [row for row in fake.records if row["stage"] == "h20"]

    assert public["status"] == "OPPORTUNITY_QUALIFIED"
    assert public["training_authorized"] is True
    assert public["fresh_external_evidence"] is False
    assert public["champion_changed"] is False
    assert public["release_authorized"] is False
    assert "private_details" not in public
    assert isinstance(internal["private_details"], dict)
    assert internal["model_call_counts"] == {
        "control_calls": 24,
        "control_invalid_count": 9,
        "repair_calls": 9,
        "valid_control_repair_calls": 0,
        "total_calls": 33,
    }
    assert len(parent_records) == 24
    assert {row["case_id"] for row in h20_records} == control_invalid_ids
    assert all(row["case_id"].startswith("h21-opened-") for row in fake.records)
    assert all("gold_method_id" not in json.dumps(row["messages"]) for row in fake.records)
    assert all("source_id" not in json.dumps(row["messages"]) for row in fake.records)
    assert fake.events.index("close:parent") < fake.events.index("load:h20")
    opening_path = (
        tmp_path
        / "artifacts/evolve/opened-source-residual-repair-v1/opportunity-opened.json"
    )
    opening = json.loads(opening_path.read_text(encoding="utf-8"))
    assert opening["model_output_opened"] is True
    assert opening["training_authorized"] is False
    assert opening["case_count"] == 24
    opening_bytes = opening_path.read_bytes()

    class ForbiddenCaller:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("terminal replay must not construct a model caller")

    monkeypatch.setattr(h21, "_AgentCaller", ForbiddenCaller)
    monkeypatch.setattr(
        h21,
        "run_exact_two_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal replay must not run model evaluation")
        ),
    )
    replay = h21.run_opened_source_residual_opportunity(config)
    assert replay == public
    assert report_path.read_bytes() == terminal_bytes
    assert opening_path.read_bytes() == opening_bytes


def test_all_valid_controls_make_zero_h20_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _copy_snapshots(tmp_path)
    master = _master()
    screen = _screen(master)
    _patch_seals(monkeypatch, master, screen)
    cases, _ = h21._extract_cases_and_manifest(config)
    fake = _install_fake_callers(
        monkeypatch,
        cases,
        control_invalid_ids=set(),
        residual_ids=set(),
    )

    public = h21.run_opened_source_residual_opportunity(config)

    assert public["status"] == "INCONCLUSIVE_OPPORTUNITY"
    assert public["training_authorized"] is False
    assert public["model_call_counts"] == {
        "control_calls": 24,
        "control_invalid_count": 0,
        "repair_calls": 0,
        "valid_control_repair_calls": 0,
        "total_calls": 24,
    }
    assert [row["stage"] for row in fake.records] == ["parent"] * 24
    assert "load:h20" not in fake.events


def test_terminal_replay_rejects_resigned_private_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _copy_snapshots(tmp_path)
    master = _master()
    screen = _screen(master)
    _patch_seals(monkeypatch, master, screen)
    cases, _ = h21._extract_cases_and_manifest(config)
    residual_ids = _qualified_residual_ids(cases)
    fake = _install_fake_callers(
        monkeypatch,
        cases,
        control_invalid_ids=residual_ids,
        residual_ids=residual_ids,
    )
    h21.run_opened_source_residual_opportunity(config)
    assert fake.records

    report_path = (
        tmp_path / "artifacts/evolve/opened-source-residual-repair-v1/report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["private_details"]["candidate"][0]["source_id"] = "tampered-source"
    report["private_details_fingerprint"] = canonical_hash(report["private_details"])
    report["result_fingerprint"] = h21._terminal_fingerprint(report)
    write_json(report_path, report)

    class ForbiddenCaller:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("tampered replay must fail before caller construction")

    monkeypatch.setattr(h21, "_AgentCaller", ForbiddenCaller)
    with pytest.raises(RuntimeError, match="candidate details changed from the frozen cases"):
        h21.run_opened_source_residual_opportunity(config)
