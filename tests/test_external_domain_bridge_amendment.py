import numpy as np

from charlie_alpha.stats_external_domain_bridge_amendment import _fixed_augmented_bank


def test_fixed_augmented_bank_reconstructs_normalized_view() -> None:
    base = {
        "vectors": np.asarray([[1.0, 0.0]], dtype=np.float64),
        "center": np.asarray([[0.0, 0.0]], dtype=np.float64),
    }
    bank = _fixed_augmented_bank(base, np.asarray([[0.0, 2.0]], dtype=np.float64))
    assert bank["vectors"].shape == (2, 2)
    assert bank["normalized"].shape == (2, 2)
    assert np.allclose(bank["normalized"], np.eye(2))
