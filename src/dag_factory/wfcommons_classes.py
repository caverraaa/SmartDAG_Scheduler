"""Deterministic WfFormat task-name -> TaskClass mapping (TZ §4, §7).

WfFormat has no task category; ``name`` is the only class signal. We use curated
per-recipe tables on the abstract task name, with a deterministic hash fallback so
unknown recipes/versions never KeyError or silently collapse to one class.
Both the tables and the fallback are disclosed in the thesis.
"""

import hashlib
import re

from src.core.task import TaskClass

_CLASSES = list(TaskClass)
_SUFFIX_RE = re.compile(r"(_(ID)?\d+)+$")


def abstract_name(name: str) -> str:
    """Strip trailing ``_<digits>`` / ``_ID<digits>`` groups (e.g. mProject_00000001)."""
    return _SUFFIX_RE.sub("", name)


def stable_hash(text: str) -> int:
    """Deterministic, cross-run hash (Python's built-in hash is salted)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


# Curated per-recipe tables. Montage names are confirmed; cybershake/blast are
# best-effort and verified/extended against generated JSON during benchmark
# regeneration (Task 6). Unknown names fall back deterministically, so an
# out-of-date table degrades gracefully rather than failing.
RECIPE_TABLES: dict[str, dict[str, TaskClass]] = {
    "montage": {
        "mProject": TaskClass.DATA_PARALLEL,
        "mDiffFit": TaskClass.STREAMING,
        "mConcatFit": TaskClass.SEQUENTIAL,
        "mBgModel": TaskClass.SEQUENTIAL,
        "mBackground": TaskClass.DATA_PARALLEL,
        "mImgtbl": TaskClass.SEQUENTIAL,
        "mAdd": TaskClass.DATA_PARALLEL,
        "mShrink": TaskClass.STREAMING,
        "mViewer": TaskClass.SEQUENTIAL,
    },
    "cybershake": {
        "PreCVM": TaskClass.SEQUENTIAL,
        "ExtractSGT": TaskClass.STREAMING,
        "SeismogramSynthesis": TaskClass.DATA_PARALLEL,
        "PeakValCalcOkaya": TaskClass.DATA_PARALLEL,
        "ZipSeis": TaskClass.SEQUENTIAL,
        "ZipPSA": TaskClass.SEQUENTIAL,
    },
    "blast": {
        "blastall": TaskClass.DATA_PARALLEL,
        "split": TaskClass.SEQUENTIAL,
        "cat": TaskClass.SEQUENTIAL,
    },
}


def infer_recipe(abstract_names: list[str]) -> str | None:
    """Return the recipe key whose table matches the most abstract names, or None."""
    best_key: str | None = None
    best_hits = 0
    for key, table in RECIPE_TABLES.items():
        hits = sum(1 for a in abstract_names if a in table)
        if hits > best_hits:
            best_key, best_hits = key, hits
    return best_key


def assign_task_class(name: str, table: dict[str, TaskClass]) -> TaskClass:
    """Curated lookup on the abstract name; deterministic hash fallback otherwise."""
    abstract = abstract_name(name)
    if abstract in table:
        return table[abstract]
    return _CLASSES[stable_hash(abstract) % 3]
