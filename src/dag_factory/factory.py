"""DAGFactory: Factory pattern over interchangeable DAG sources (TZ §3, §7)."""

import json

import numpy as np

from src.core.dag import TaskDAG
from src.dag_factory.synthetic import generate_synthetic
from src.dag_factory.wfcommons_adapter import parse_wfformat
from src.dag_factory.wfcommons_config import load_wfcommons_params


class DAGFactory:
    @classmethod
    def create(cls, source: str, rng: np.random.Generator, **params: object) -> TaskDAG:
        if source == "synthetic":
            return generate_synthetic(
                rng,
                n_tasks=int(params["n_tasks"]),  # type: ignore[arg-type]
                n_layers=int(params["n_layers"]),  # type: ignore[arg-type]
                edge_prob=float(params["edge_prob"]),  # type: ignore[arg-type]
                ccr=float(params["ccr"]),  # type: ignore[arg-type]
            )
        if source == "wfcommons":
            recipe = params.get("recipe")
            return cls.load_from_wfcommons(
                str(params["path"]),
                rng,
                recipe=str(recipe) if recipe is not None else None,
            )
        raise ValueError(f"Unknown DAG source: {source!r}")

    @classmethod
    def load_from_wfcommons(
        cls,
        path: str,
        rng: np.random.Generator,
        recipe: str | None = None,
        config_path: str = "config.yaml",
    ) -> TaskDAG:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        params = load_wfcommons_params(config_path)
        return parse_wfformat(doc, rng, params, recipe=recipe)
