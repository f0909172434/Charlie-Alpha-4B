from charlie_alpha.stats_router_reduced import _reduced_mapping


def test_reduced_mapping_excludes_only_registered_family() -> None:
    mapping = {
        "a": {"slug": "expert-a", "checkpoint_name": "update-01", "update": 1},
        "b": {"slug": "expert-b", "checkpoint_name": "update-02", "update": 2},
    }
    reduced = _reduced_mapping(mapping, "a")
    assert reduced["a"]["slug"] == "parent"
    assert reduced["a"]["checkpoint_name"] == "parent"
    assert reduced["a"]["prospective_exclusion"] is True
    assert reduced["b"] == mapping["b"]
    assert mapping["a"]["slug"] == "expert-a"
