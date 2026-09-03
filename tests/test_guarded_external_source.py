from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from charlie_alpha import stats_guarded_external_source as source_module
from charlie_alpha.config import ProjectConfig, load_config
from charlie_alpha.io_utils import canonical_hash, sha256_file, write_json, write_jsonl
from charlie_alpha.stats_guarded_external import (
    _verify_child_contract,
    _verify_selected_data,
)
from charlie_alpha.stats_guarded_external_source import (
    _METHOD_IDS,
    _SOURCES,
    _bioc_document,
    _method_cells,
    _source_summary,
    prepare_guarded_external_master_contract,
)

_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOTS = (
    _ROOT / "data" / "evolve" / "guarded-external-v1" / "source-screen" / "snapshots"
)
_GATES = {
    "minimum_eligible_cases": 150,
    "minimum_distinct_methods": 8,
    "minimum_coverage_fraction": 0.80,
    "maximum_single_method_fraction": 0.40,
}


def _cells(pmcid: str) -> list[dict]:
    source = _SOURCES[pmcid]
    payload = (_SNAPSHOTS / f"{pmcid}.bioc.json").read_bytes()
    document = _bioc_document(
        payload,
        pmcid=pmcid,
        expected_license=str(source["license"]),
        expected_doi=str(source["doi"]),
        expected_title=str(source["title"]),
    )
    return _method_cells(pmcid=pmcid, source=source, document=document)


def test_screened_source_counts_use_the_actual_28_method_catalog() -> None:
    assert len(_METHOD_IDS) == 28
    assert len(set(_METHOD_IDS)) == 28
    assert all(
        method_id in _METHOD_IDS
        for source in _SOURCES.values()
        for method_id in source["aliases"].values()
    )
    expected = {
        "PMC8327789": (18, 6, 6),
        "PMC8483143": (18, 7, 7),
        "PMC6639881": (27, 9, 8),
        "PMC2996580": (11, 4, 4),
    }
    all_cells: list[dict] = []
    for pmcid, (cell_count, eligible_count, distinct_count) in expected.items():
        cells = _cells(pmcid)
        all_cells.extend(cells)
        eligible = [row for row in cells if row["mapped_method_id"] is not None]
        assert len(cells) == cell_count
        assert len(eligible) == eligible_count
        assert len({row["mapped_method_id"] for row in eligible}) == distinct_count

        summary = _source_summary(
            pmcid=pmcid,
            source=_SOURCES[pmcid],
            cells=cells,
            gates=_GATES,
        )
        assert summary["qualified"] is False
        assert summary["checks"]["minimum_eligible_cases"] is False
        assert summary["checks"]["minimum_coverage_fraction"] is False
        assert summary["checks"]["source_blind"] is False

    aggregate_eligible = [row for row in all_cells if row["mapped_method_id"] is not None]
    assert len(all_cells) == 74
    assert len(aggregate_eligible) == 26
    assert len({row["mapped_method_id"] for row in aggregate_eligible}) == 11


def test_screen_preserves_source_spelling_and_rejects_ambiguous_or_absent_methods() -> None:
    cells_848 = _cells("PMC8483143")
    wilcox = next(row for row in cells_848 if row["source_label"].startswith("Wilcox on"))
    assert wilcox["source_label"] == "Wilcox on signed-rank test"
    assert wilcox["mapped_method_id"] == "wilcoxon_signed_rank"

    cells_832 = _cells("PMC8327789")
    kruskal = next(row for row in cells_832 if row["source_label"].startswith("Kruskal"))
    assert kruskal["mapped_method_id"] is None

    cells_299 = _cells("PMC2996580")
    student = next(row for row in cells_299 if row["source_label"] == "Student’s t-test")
    assert "paired or unpaired" in student["condition"]
    assert student["mapped_method_id"] is None


def test_snapshot_files_are_valid_json() -> None:
    for pmcid in _SOURCES:
        path = _SNAPSHOTS / f"{pmcid}.bioc.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(value, list) and len(value) == 1
        assert sha256_file(path) == _SOURCES[pmcid]["snapshot_sha256"]


def test_source_screen_gates_match_the_frozen_yaml() -> None:
    config = load_config(_ROOT / "configs" / "pipeline.evolve.yaml")

    assert config.section("guarded_external")["source_gates"] == _GATES


def _temporary_master_config(tmp_path: Path) -> ProjectConfig:
    frozen = load_config(_ROOT / "configs" / "pipeline.evolve.yaml")
    values = {
        "paths": {
            "artifact_dir": "artifacts",
            "evolution_dir": "data/evolve",
        },
        "guarded_external": deepcopy(frozen.section("guarded_external")),
    }
    return ProjectConfig(
        path=tmp_path / "configs" / "pipeline.evolve.yaml",
        root=tmp_path,
        values=values,
        sources={},
    )


def test_master_lock_reopens_identically_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = {
        "control": {"adapter_sha256": "control"},
        "repair": {"adapter_sha256": "repair"},
        "decoding": {"max_tokens": 160, "temperature": 0.0, "top_p": 1.0},
    }
    runtime["fingerprint"] = canonical_hash(runtime)
    monkeypatch.setattr(source_module, "_runtime_manifest", lambda _config: deepcopy(runtime))
    monkeypatch.setattr(
        source_module,
        "_implementation_manifest",
        lambda: {"stats_guarded_external.py": "implementation-sha256"},
    )
    config = _temporary_master_config(tmp_path)

    first = prepare_guarded_external_master_contract(config)
    lock_path = tmp_path / "artifacts" / "guarded-external-v1" / "master-contract.json"
    locked_bytes = lock_path.read_bytes()
    second = prepare_guarded_external_master_contract(config)

    assert second == first
    assert lock_path.read_bytes() == locked_bytes

    drifted_runtime = deepcopy(runtime)
    drifted_runtime["decoding"]["max_tokens"] = 161
    drifted_runtime["fingerprint"] = canonical_hash(
        {key: value for key, value in drifted_runtime.items() if key != "fingerprint"}
    )
    monkeypatch.setattr(
        source_module, "_runtime_manifest", lambda _config: deepcopy(drifted_runtime)
    )
    with pytest.raises(RuntimeError, match="master contract is immutable"):
        prepare_guarded_external_master_contract(config)
    assert lock_path.read_bytes() == locked_bytes


def _qualified_selected_fixture(tmp_path: Path) -> tuple[dict, dict, dict, Path]:
    selected_root = tmp_path / "selected-source"
    selected_root.mkdir()
    snapshot_path = selected_root / "source.snapshot"
    snapshot_path.write_bytes(b"future-source-snapshot")
    alias_map = {"paired t": "paired_t"}
    extraction = {"version": 1, "complete_frame": True}
    source = {
        "source_id": "future-source-v1",
        "stable_id": "future-source-v1",
        "license": "CC BY 4.0",
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "alias_map": alias_map,
        "alias_map_fingerprint": canonical_hash(alias_map),
        "extraction_contract": extraction,
        "extraction_contract_fingerprint": canonical_hash(extraction),
    }
    runtime = {"decoding": {"max_tokens": 160, "temperature": 0.0, "top_p": 1.0}}
    runtime["fingerprint"] = canonical_hash(runtime)
    master = {
        "runtime": runtime,
        "implementation_sha256": {"runtime.py": "sha256"},
        "source_protocol": {
            "qualification_gates": deepcopy(_GATES),
            "opened_source_exclusions": {},
        },
    }
    master["fingerprint"] = canonical_hash(master)
    contract = {
        "master_contract_fingerprint": master["fingerprint"],
        "runtime_fingerprint": runtime["fingerprint"],
        "implementation_sha256": deepcopy(master["implementation_sha256"]),
        "source_selected_before_opening": True,
        "source_qualified": True,
        "source_qualified_not_blind": False,
        "source": source,
    }
    contract["fingerprint"] = canonical_hash(contract)

    method_ids = list(_METHOD_IDS[:8])
    cases = [
        {
            "case_id": f"future-{index:03d}",
            "source_id": source["source_id"],
            "question": f"Future question {index}",
            "gold_method_id": method_ids[index % len(method_ids)],
        }
        for index in range(150)
    ]
    cases_path = selected_root / "cases.jsonl"
    write_jsonl(cases_path, cases)
    frame = [
        {
            "frame_id": f"frame-{index:03d}",
            "case_id": case["case_id"],
            "mapped_method_id": case["gold_method_id"],
        }
        for index, case in enumerate(cases)
    ]
    frame.extend(
        {
            "frame_id": f"unmapped-{index:03d}",
            "case_id": None,
            "mapped_method_id": None,
        }
        for index in range(10)
    )
    frame_path = selected_root / "complete-frame.jsonl"
    write_jsonl(frame_path, frame)
    overlap = {
        "overlap_count": 0,
        "case_fingerprint": canonical_hash(cases),
        "normalization_fingerprint": "normalization-sha256",
        "historical_corpus_fingerprint": "history-sha256",
    }
    overlap["fingerprint"] = canonical_hash(overlap)
    overlap_path = selected_root / "overlap.json"
    write_json(overlap_path, overlap)
    data = {
        "contract_fingerprint": contract["fingerprint"],
        "complete": True,
        "evaluation_authorized": True,
        "cases_path": str(cases_path),
        "cases_sha256": sha256_file(cases_path),
        "case_fingerprint": canonical_hash(cases),
        "case_count": len(cases),
        "complete_frame_path": str(frame_path),
        "complete_frame_sha256": sha256_file(frame_path),
        "complete_frame_fingerprint": canonical_hash(frame),
        "alias_map_fingerprint": source["alias_map_fingerprint"],
        "extraction_contract_fingerprint": source["extraction_contract_fingerprint"],
        "overlap_manifest_path": str(overlap_path),
        "overlap_manifest_sha256": sha256_file(overlap_path),
    }
    data["data_fingerprint"] = canonical_hash(data)
    return master, contract, data, selected_root


def _resign_data(data: dict) -> None:
    data["data_fingerprint"] = canonical_hash(
        {key: value for key, value in data.items() if key != "data_fingerprint"}
    )


def test_selected_source_receipts_accept_a_complete_qualified_fixture(tmp_path: Path) -> None:
    master, contract, data, selected_root = _qualified_selected_fixture(tmp_path)

    _verify_child_contract(contract, master)
    cases = _verify_selected_data(data, contract, master, selected_root=selected_root)

    assert len(cases) == 150


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("source-id", "exactly the selected source"),
        ("snapshot", "snapshot changed"),
        ("frame", "frame and materialized cases changed"),
        ("overlap", "overlaps historical evidence"),
        ("concentration", "qualification changed"),
    ],
)
def test_selected_source_receipts_reject_resigned_evidence_drift(
    tmp_path: Path, drift: str, message: str
) -> None:
    master, contract, data, selected_root = _qualified_selected_fixture(tmp_path)
    cases_path = Path(data["cases_path"])
    frame_path = Path(data["complete_frame_path"])
    overlap_path = Path(data["overlap_manifest_path"])

    if drift == "snapshot":
        Path(contract["source"]["snapshot_path"]).write_bytes(b"changed")
    elif drift == "frame":
        frame = [json.loads(line) for line in frame_path.read_text().splitlines()]
        frame[0]["case_id"] = "other-case"
        write_jsonl(frame_path, frame)
        data["complete_frame_sha256"] = sha256_file(frame_path)
        data["complete_frame_fingerprint"] = canonical_hash(frame)
        _resign_data(data)
    else:
        cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
        if drift == "source-id":
            cases[0]["source_id"] = "other-source"
        elif drift == "concentration":
            for case in cases:
                case["gold_method_id"] = "paired_t"
            frame = [json.loads(line) for line in frame_path.read_text().splitlines()]
            by_case = {case["case_id"]: case["gold_method_id"] for case in cases}
            for row in frame:
                if row["case_id"] in by_case:
                    row["mapped_method_id"] = by_case[row["case_id"]]
            write_jsonl(frame_path, frame)
            data["complete_frame_sha256"] = sha256_file(frame_path)
            data["complete_frame_fingerprint"] = canonical_hash(frame)
        elif drift == "overlap":
            overlap = json.loads(overlap_path.read_text())
            overlap["overlap_count"] = 1
            overlap["fingerprint"] = canonical_hash(
                {key: value for key, value in overlap.items() if key != "fingerprint"}
            )
            write_json(overlap_path, overlap)
            data["overlap_manifest_sha256"] = sha256_file(overlap_path)
            _resign_data(data)
            with pytest.raises(RuntimeError, match=message):
                _verify_selected_data(data, contract, master, selected_root=selected_root)
            return
        write_jsonl(cases_path, cases)
        data["cases_sha256"] = sha256_file(cases_path)
        data["case_fingerprint"] = canonical_hash(cases)
        overlap = json.loads(overlap_path.read_text())
        overlap["case_fingerprint"] = data["case_fingerprint"]
        overlap["fingerprint"] = canonical_hash(
            {key: value for key, value in overlap.items() if key != "fingerprint"}
        )
        write_json(overlap_path, overlap)
        data["overlap_manifest_sha256"] = sha256_file(overlap_path)
        _resign_data(data)

    with pytest.raises(RuntimeError, match=message):
        _verify_selected_data(data, contract, master, selected_root=selected_root)
