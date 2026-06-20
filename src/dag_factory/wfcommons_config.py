"""Modeling constants for the WfCommons parser, loaded from config.yaml (TZ §5, §7)."""

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class WfcommonsParams:
    default_mem: float
    eps: float
    bytes_to_unit: float
    mem_min: float
    mem_max: float
    memory_ref_bytes: float


def load_wfcommons_params(path: str = "config.yaml") -> WfcommonsParams:
    """Parse the ``wfcommons:`` block of config.yaml into a typed, frozen params object."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    wf = raw["wfcommons"]
    return WfcommonsParams(
        default_mem=float(wf["default_mem"]),
        eps=float(wf["eps"]),
        bytes_to_unit=float(wf["bytes_to_unit"]),
        mem_min=float(wf["mem_min"]),
        mem_max=float(wf["mem_max"]),
        memory_ref_bytes=float(wf["memory_ref_bytes"]),
    )
