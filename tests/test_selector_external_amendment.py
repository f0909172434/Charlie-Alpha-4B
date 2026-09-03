from charlie_alpha.stats_selector_external_amendment import _predicted_method


def test_e3_amended_parser_accepts_h14_methods_array() -> None:
    predicted, valid = _predicted_method('{"methods":["paired_t"],"columns":["before","after"]}')
    assert valid
    assert predicted == "paired_t"


def test_e3_amended_parser_rejects_multiple_methods() -> None:
    answer = '{"methods":["paired_t","wilcoxon_signed_rank"],"columns":[]}'
    predicted, valid = _predicted_method(answer)
    assert not valid
    assert predicted is None
