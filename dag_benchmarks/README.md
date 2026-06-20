# dag_benchmarks

Frozen, committed WfFormat v1.5 workflows for the M5 evaluation grid. Loaded via
`DAGFactory.load_from_wfcommons(path, rng, recipe=...)`. These are DATA, not
generated at test time. **The committed JSON files are authoritative** — they
freeze the exact instances every strategy is evaluated on.

Generated with **wfcommons 1.4**. Recipes: **Montage / Genome / Blast**.
CyberShake is not a bundled wfchef recipe in wfcommons 1.4, so Genome takes its
place (Genome's `individuals_ID...` task naming is also what the abstract-name
parser is tuned for). wfchef cannot shrink a recipe below its base
microstructure, so 60 is the practical minimum (each yields ~58 tasks).

Committed set (seed 42):

| file | tasks | edges | task classes |
|------|------:|------:|--------------|
| `montage_60_seed42.json` | 58 | 114 | data_parallel / streaming / sequential |
| `genome_60_seed42.json`  | 58 |  86 | data_parallel / sequential / streaming |
| `blast_60_seed42.json`   | 58 | 165 | data_parallel / sequential |

## Regenerating (requires `pip install wfcommons`)

Pinned seed = 42. Exact commands:

    .venv/bin/python -m tools.gen_wfcommons --recipe montage --n-tasks 60 --seed 42
    .venv/bin/python -m tools.gen_wfcommons --recipe genome  --n-tasks 60 --seed 42
    .venv/bin/python -m tools.gen_wfcommons --recipe blast   --n-tasks 60 --seed 42

**Reproducibility:** the seed pins the DAG *structure* (tasks, runtimes, file
sizes, dependencies). wfcommons names files with `uuid4` and stamps `createdAt`,
both of which ignore the seed — so regenerated JSON is **not byte-identical**.
However, file IDs only matter relationally (producer↔consumer overlap), so the
**parsed `TaskDAG` is identical** across regenerations (verified: base_cost,
task_class, mem_required, and edge volumes all match). Treat the committed files
as the frozen artifacts; regeneration yields a structurally-equivalent set.

After regenerating, run the drift check (which also confirms the curated class
tables in `src/dag_factory/wfcommons_classes.py` still match the real task names):

    .venv/bin/pytest -m wfcommons
