# dag_benchmarks

Frozen, committed WfFormat v1.5 workflows for the M5 evaluation grid. Loaded via
`DAGFactory.load_from_wfcommons(path, rng, recipe=...)`. These are DATA, not
generated at test time.

## Regenerating (requires `pip install wfcommons`)

Pinned seed = 42. Exact commands:

    .venv/bin/python -m tools.gen_wfcommons --recipe montage    --n-tasks 30 --seed 42
    .venv/bin/python -m tools.gen_wfcommons --recipe cybershake --n-tasks 30 --seed 42
    .venv/bin/python -m tools.gen_wfcommons --recipe blast      --n-tasks 30 --seed 42

After regenerating, run the drift check and verify/extend the curated class tables
in `src/dag_factory/wfcommons_classes.py` against the real task names:

    .venv/bin/pytest -m wfcommons
