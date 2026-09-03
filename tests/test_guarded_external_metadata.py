from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from charlie_alpha.config import load_config
from charlie_alpha.stats_guarded_external_metadata import (
    _EXPECTED_RECEIPT_SHA256,
    _candidate_summaries,
    _validate_receipts,
    prepare_guarded_external_future_metadata_screen,
)

_ROOT = Path(__file__).resolve().parents[1]
_RECEIPT_PATH = (
    _ROOT
    / "data"
    / "evolve"
    / "guarded-external-v1"
    / "future-source-metadata"
    / "metadata-receipts.json"
)


def _payload() -> dict:
    return json.loads(_RECEIPT_PATH.read_text(encoding="utf-8"))


def test_metadata_receipts_are_content_endpoint_free_and_complete() -> None:
    payload = _payload()
    index = _validate_receipts(payload)

    assert payload["receipt_count"] == 87
    assert len(index) == 87
    assert all(value is False for value in payload["content_access_policy"].values())
    assert (
        __import__("hashlib").sha256(_RECEIPT_PATH.read_bytes()).hexdigest()
        == _EXPECTED_RECEIPT_SHA256
    )


def test_metadata_screen_rejects_a_rows_endpoint_even_when_receipt_is_resigned() -> None:
    payload = _payload()
    drifted = deepcopy(payload)
    drifted["receipts"][0]["url"] = "https://datasets-server.huggingface.co/rows"

    with pytest.raises(RuntimeError, match="content endpoint"):
        _validate_receipts(drifted)


def test_all_metadata_candidates_fail_before_child_selection() -> None:
    candidates = _candidate_summaries(_validate_receipts(_payload()))

    assert len(candidates) == 24
    assert sum(bool(candidate["near_candidate"]) for candidate in candidates) == 7
    assert all(candidate["disposition"] == "metadata-only-reject" for candidate in candidates)
    assert all(
        not candidate["qualification_checks"]["qualified_child_contract_freezable"]
        for candidate in candidates
    )

    scholar = next(
        candidate
        for candidate in candidates
        if candidate["source_id"] == "ofrencber/scholargate-research-methods"
    )
    assert scholar["revision"] == "e6250d756656158d890d8f4a6fda00f705bb29e6"
    assert scholar["metadata_facts"]["processed_row_count"] == 6620
    methods_file = next(
        item
        for item in scholar["metadata_facts"]["pinned_tree_files"]
        if item["path"] == "methods.jsonl"
    )
    assert methods_file["lfs_sha256"] == (
        "12927edae02c5bbc97671c8299a22646617a4d06683e90b90617d4b4161a0b77"
    )


def test_future_metadata_screen_replays_without_mutating_its_lock() -> None:
    config = load_config(_ROOT / "configs" / "pipeline.evolve.yaml")
    first = prepare_guarded_external_future_metadata_screen(config)
    lock_path = (
        _ROOT
        / "artifacts"
        / "evolve"
        / "guarded-external-v1"
        / "future-source-metadata-screen.json"
    )
    before = lock_path.read_bytes()
    second = prepare_guarded_external_future_metadata_screen(config)

    assert first == second
    assert lock_path.read_bytes() == before
    assert first["status"] == "NO_SOURCE_SELECTED_METADATA_SCREEN"
    assert first["source_selected"] is False
    assert first["evaluation_authorized"] is False
    assert first["dataset_rows_opened"] is False
