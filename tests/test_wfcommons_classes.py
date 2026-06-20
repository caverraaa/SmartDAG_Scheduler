from src.core.task import TaskClass
from src.dag_factory.wfcommons_classes import (
    RECIPE_TABLES,
    abstract_name,
    assign_task_class,
    infer_recipe,
    stable_hash,
)


def test_abstract_name_strips_numeric_and_id_suffixes() -> None:
    assert abstract_name("mProject_00000001") == "mProject"
    assert abstract_name("individuals_ID0000007_0") == "individuals"
    assert abstract_name("mConcatFit") == "mConcatFit"


def test_stable_hash_is_deterministic_and_nonnegative() -> None:
    assert stable_hash("mProject") == stable_hash("mProject")
    assert stable_hash("mProject") >= 0


def test_curated_lookup_uses_table() -> None:
    table = RECIPE_TABLES["montage"]
    assert assign_task_class("mProject_00000001", table) is TaskClass.DATA_PARALLEL
    assert assign_task_class("mConcatFit_00000001", table) is TaskClass.SEQUENTIAL


def test_unmatched_name_uses_deterministic_fallback() -> None:
    table = RECIPE_TABLES["montage"]
    classes = list(TaskClass)
    expected = classes[stable_hash("totallyUnknownTask") % 3]
    assert assign_task_class("totallyUnknownTask_42", table) is expected


def test_every_table_spans_at_least_two_classes() -> None:
    for key, table in RECIPE_TABLES.items():
        assert len(set(table.values())) >= 2, f"{key} collapses to one class"


def test_infer_recipe_picks_best_match() -> None:
    names = ["mProject", "mDiffFit", "mConcatFit", "unknown"]
    assert infer_recipe(names) == "montage"
    assert infer_recipe(["nothing", "matches"]) is None
