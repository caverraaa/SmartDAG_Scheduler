"""Optional CLI: generate WfFormat JSON benchmarks via wfcommons (TZ §7, §11).

wfcommons is an OPTIONAL extra; it is imported lazily here only. Runtime and the
default test suite never import it. Run:
    .venv/bin/python -m tools.gen_wfcommons --recipe montage --n-tasks 20 --seed 42
"""

import argparse
import os

_RECIPES = {
    "montage": "MontageRecipe",
    "cybershake": "CyberShakeRecipe",
    "blast": "BlastRecipe",
}


def _load_recipe_class(recipe: str):
    try:
        from wfcommons import recipes  # noqa: PLC0415  (lazy, optional dependency)
    except ImportError as exc:  # pragma: no cover - exercised only without wfcommons
        raise ImportError(
            "wfcommons is required to generate benchmarks: pip install wfcommons"
        ) from exc
    if recipe not in _RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}; choices: {sorted(_RECIPES)}")
    return getattr(recipes, _RECIPES[recipe])


def generate(recipe: str, n_tasks: int, seed: int, out_dir: str) -> str:
    """Build one workflow and write it as WfFormat JSON; return the written path."""
    import random

    random.seed(seed)
    recipe_cls = _load_recipe_class(recipe)
    workflow = recipe_cls.from_num_tasks(n_tasks).build_workflow()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{recipe}_{n_tasks}_seed{seed}.json")
    workflow.write_json(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WfFormat JSON benchmarks.")
    parser.add_argument("--recipe", required=True, choices=sorted(_RECIPES))
    parser.add_argument("--n-tasks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="dag_benchmarks")
    args = parser.parse_args()
    path = generate(args.recipe, args.n_tasks, args.seed, args.out_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
