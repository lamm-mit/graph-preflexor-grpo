"""Dataset loading, auditing, grouped validation, and deterministic sampling."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from graph_completion_metrics import compute_graph_metrics, mode_primary_score
from graph_completion_parsing import (
    GRAPH_COMPLETION_MODES,
    canonicalize_graph,
    graph_pair_contract_errors,
    parse_graph_json,
)


DEFAULT_DATASET = "lamm-mit/graph-canvas-inpainting-121k"
REQUIRED_COLUMNS = {
    "source_index",
    "variant_index",
    "mode",
    "condition",
    "x0",
    "fixed_node_ids",
    "fixed_edge_keys",
    "x0_graph_json",
    "x1_graph_json",
}


@dataclass
class DatasetAudit:
    rows_before: int
    rows_after: int
    invalid_by_mode: dict[str, int]
    examples: list[str]


@dataclass
class PreparedDatasets:
    train: Any
    validation: Any
    test: Any
    validation_source_indices: list[str]
    train_audit: DatasetAudit
    test_audit: DatasetAudit


def _stable_key(value: Any, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def load_graph_completion_dataset(dataset: str = DEFAULT_DATASET) -> Any:
    """Load a Hub dataset or a local datasets.load_from_disk directory."""

    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    path = Path(dataset).expanduser()
    loaded = load_from_disk(str(path)) if path.is_dir() else load_dataset(dataset)
    if isinstance(loaded, Dataset):
        raise ValueError(
            "Graph completion requires the official train/test split; local data must be a DatasetDict."
        )
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"unsupported dataset object: {type(loaded).__name__}")
    missing_splits = {"train", "test"} - set(loaded)
    if missing_splits:
        raise ValueError(f"dataset is missing official splits: {sorted(missing_splits)}")
    for split_name in ("train", "test"):
        missing = REQUIRED_COLUMNS - set(loaded[split_name].column_names)
        if missing:
            raise ValueError(f"{split_name} is missing columns: {sorted(missing)}")
    return loaded


def audit_source_split(train: Any, test: Any) -> dict[str, int]:
    train_sources = {str(value) for value in train["source_index"]}
    test_sources = {str(value) for value in test["source_index"]}
    overlap = train_sources & test_sources
    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(
            f"official source-level split leaks {len(overlap)} source_index values; examples={examples}"
        )
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_sources": len(train_sources),
        "test_sources": len(test_sources),
        "overlap_sources": 0,
    }


def audit_and_filter_pairs(
    dataset: Any,
    *,
    split_name: str,
    invalid_pair_policy: str = "filter",
    max_examples: int = 5,
) -> tuple[Any, DatasetAudit]:
    policy = invalid_pair_policy.strip().lower()
    if policy not in {"filter", "error"}:
        raise ValueError("invalid_pair_policy must be 'filter' or 'error'")
    invalid_indices: list[int] = []
    counts: Counter[str] = Counter()
    examples: list[str] = []
    for index, row in enumerate(dataset):
        errors = graph_pair_contract_errors(row)
        if not errors:
            continue
        invalid_indices.append(index)
        mode = str(row.get("mode", "unknown"))
        counts[mode] += 1
        if len(examples) < max_examples:
            examples.append(
                f"{split_name}[{index}] source_index={row.get('source_index')} "
                f"variant_index={row.get('variant_index')} mode={mode}: {'; '.join(errors)}"
            )
    if invalid_indices:
        message = (
            f"{split_name}: found {len(invalid_indices)} invalid graph pair(s); "
            f"by_mode={dict(sorted(counts.items()))}; examples={examples}"
        )
        if policy == "error":
            raise ValueError(message)
        print(f"[graph-completion data] {message}")
        invalid = set(invalid_indices)
        dataset = dataset.select([index for index in range(len(dataset)) if index not in invalid])
    else:
        print(f"[graph-completion data] {split_name}: 0 invalid graph pairs")
    return dataset, DatasetAudit(
        rows_before=len(dataset) + len(invalid_indices),
        rows_after=len(dataset),
        invalid_by_mode=dict(sorted(counts.items())),
        examples=examples,
    )


def create_or_load_validation_manifest(
    train: Any,
    manifest_path: str,
    *,
    seed: int = 42,
    source_count: int = 512,
    source_fraction: Optional[float] = None,
) -> list[str]:
    """Persist a deterministic validation subset made only of whole source groups."""

    path = Path(manifest_path)
    available = {str(value) for value in train["source_index"]}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = [str(value) for value in payload["validation_source_indices"]]
        missing = set(selected) - available
        if missing:
            raise ValueError(
                f"validation manifest contains {len(missing)} unavailable train sources"
            )
        return selected
    ordered = sorted(available, key=lambda value: _stable_key(value, seed))
    if source_fraction is not None:
        if not 0.0 < source_fraction < 1.0:
            raise ValueError("validation_source_fraction must be between 0 and 1")
        count = max(1, round(len(ordered) * source_fraction))
    else:
        count = max(1, min(source_count, max(1, len(ordered) - 1)))
    selected = ordered[:count]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "validation_source_indices": selected,
        "available_train_source_count": len(ordered),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return selected


def _select_sources(dataset: Any, sources: set[str], *, include: bool) -> Any:
    indices = [
        index
        for index, value in enumerate(dataset["source_index"])
        if (str(value) in sources) == include
    ]
    return dataset.select(indices)


def _filter_modes(dataset: Any, modes: Optional[Iterable[str]]) -> Any:
    if not modes:
        return dataset
    selected = {str(mode) for mode in modes}
    unknown = selected - set(GRAPH_COMPLETION_MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    return dataset.select(
        [index for index, mode in enumerate(dataset["mode"]) if str(mode) in selected]
    )


def _cap_sources(dataset: Any, max_sources: Optional[int], seed: int) -> Any:
    if not max_sources or max_sources <= 0:
        return dataset
    sources = sorted(
        {str(value) for value in dataset["source_index"]},
        key=lambda value: _stable_key(value, seed),
    )[:max_sources]
    return _select_sources(dataset, set(sources), include=True)


def mode_balanced_cap(dataset: Any, max_rows: Optional[int], seed: int = 42) -> Any:
    """Choose rows by deterministic round-robin across modes."""

    if not max_rows or max_rows <= 0 or len(dataset) <= max_rows:
        return dataset
    by_mode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        by_mode[str(row["mode"])].append(index)
    for mode, indices in by_mode.items():
        indices.sort(
            key=lambda index: _stable_key(
                f"{dataset[index]['source_index']}:{dataset[index]['variant_index']}:{mode}", seed
            )
        )
    chosen: list[int] = []
    modes = [mode for mode in GRAPH_COMPLETION_MODES if mode in by_mode]
    cursor = 0
    while len(chosen) < max_rows and modes:
        mode = modes[cursor % len(modes)]
        if by_mode[mode]:
            chosen.append(by_mode[mode].pop(0))
        if not by_mode[mode]:
            modes.remove(mode)
            cursor = 0
        else:
            cursor += 1
    return dataset.select(chosen)


def add_no_edit_baselines(dataset: Any) -> Any:
    """Cache the unchanged-x0 benchmark and task-primary score per row."""

    def calculate(row: Mapping[str, Any]) -> dict[str, Any]:
        x0 = canonicalize_graph(parse_graph_json(row["x0_graph_json"]))
        target = canonicalize_graph(parse_graph_json(row["x1_graph_json"]))
        metrics = compute_graph_metrics(
            x0,
            target,
            x0,
            fixed_node_ids=list(row.get("fixed_node_ids", []) or []),
            fixed_edge_keys=list(row.get("fixed_edge_keys", []) or []),
        )
        return {
            "no_edit_primary": float(mode_primary_score(str(row["mode"]), metrics)),
            "no_edit_exact": float(metrics["exact_canonical_match"]),
        }

    return dataset.map(calculate, desc="Caching graph no-edit baselines")


def prepare_graph_completion_datasets(
    dataset_name: str,
    *,
    validation_manifest: str,
    invalid_pair_policy: str = "filter",
    seed: int = 42,
    validation_source_count: int = 512,
    validation_source_fraction: Optional[float] = None,
    modes: Optional[Iterable[str]] = None,
    max_train_rows: Optional[int] = None,
    max_eval_rows: Optional[int] = None,
    max_source_graphs: Optional[int] = None,
) -> PreparedDatasets:
    dataset = load_graph_completion_dataset(dataset_name)
    split_summary = audit_source_split(dataset["train"], dataset["test"])
    print(f"[graph-completion data] official split audit: {split_summary}")
    train, train_audit = audit_and_filter_pairs(
        dataset["train"], split_name="train", invalid_pair_policy=invalid_pair_policy
    )
    test, test_audit = audit_and_filter_pairs(
        dataset["test"], split_name="test", invalid_pair_policy=invalid_pair_policy
    )
    validation_sources = create_or_load_validation_manifest(
        train,
        validation_manifest,
        seed=seed,
        source_count=validation_source_count,
        source_fraction=validation_source_fraction,
    )
    source_set = set(validation_sources)
    validation = _select_sources(train, source_set, include=True)
    train = _select_sources(train, source_set, include=False)
    train = _filter_modes(train, modes)
    validation = _filter_modes(validation, modes)
    test = _filter_modes(test, modes)
    train = _cap_sources(train, max_source_graphs, seed)
    train = mode_balanced_cap(train, max_train_rows, seed)
    validation = mode_balanced_cap(validation, max_eval_rows, seed + 1)
    test = mode_balanced_cap(test, max_eval_rows, seed + 2)
    train = add_no_edit_baselines(train)
    validation = add_no_edit_baselines(validation)
    test = add_no_edit_baselines(test)
    print(
        "[graph-completion data] prepared "
        f"train={len(train)}, validation={len(validation)}, test={len(test)}, "
        f"validation_sources={len(validation_sources)}"
    )
    return PreparedDatasets(
        train=train,
        validation=validation,
        test=test,
        validation_source_indices=validation_sources,
        train_audit=train_audit,
        test_audit=test_audit,
    )

