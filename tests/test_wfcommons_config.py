import dataclasses

import pytest

from src.dag_factory.wfcommons_config import WfcommonsParams, load_wfcommons_params


def test_load_wfcommons_params_from_config() -> None:
    p = load_wfcommons_params("config.yaml")
    assert isinstance(p, WfcommonsParams)
    assert p.mem_min < p.mem_max
    assert p.eps > 0.0
    assert p.bytes_to_unit > 0.0
    assert p.memory_ref_bytes > 0.0


def test_wfcommons_params_is_frozen() -> None:
    p = WfcommonsParams(
        default_mem=4.0,
        eps=0.01,
        bytes_to_unit=1e-6,
        mem_min=1.0,
        mem_max=8.0,
        memory_ref_bytes=8e9,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.eps = 0.5  # type: ignore[misc]
