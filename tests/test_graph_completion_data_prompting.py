import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, DatasetDict

from graph_completion_data import (
    audit_and_filter_pairs,
    audit_source_split,
    create_or_load_validation_manifest,
    prepare_graph_completion_datasets,
)
from graph_completion_parsing import render_graph_canvas
from graph_completion_prompting import apply_graph_completion_chat_template
import graph_completion_modeling
from graph_completion_modeling import (
    build_compatible_grpo_config,
    continuous_batching_compatibility,
    load_graph_completion_model_and_tokenizer,
    model_load_kwargs,
    resolve_continuous_batching,
)
from run_grpo_graph_completion import build_parser


TARGET = {
    "nodes": [{"id": "A", "metadata": {"x": 1}}],
    "edges": [],
}
EMPTY = {"nodes": [], "edges": []}


def make_row(source, variant=0):
    return {
        "source_index": source,
        "variant_index": variant,
        "mode": "prior_empty",
        "condition": "A condition",
        "x0": render_graph_canvas(EMPTY),
        "fixed_node_ids": [],
        "fixed_edge_keys": [],
        "x0_graph_json": json.dumps(EMPTY),
        "x1_graph_json": json.dumps(TARGET),
    }


def test_source_split_audit_and_deterministic_manifest(tmp_path):
    train = Dataset.from_list([make_row(index) for index in range(10)])
    test = Dataset.from_list([make_row(index) for index in range(10, 13)])
    summary = audit_source_split(train, test)
    assert summary["overlap_sources"] == 0
    manifest = tmp_path / "validation.json"
    first = create_or_load_validation_manifest(train, str(manifest), seed=7, source_count=3)
    second = create_or_load_validation_manifest(train, str(manifest), seed=999, source_count=8)
    assert first == second
    assert len(first) == 3


def test_source_split_leakage_fails():
    with pytest.raises(ValueError, match="leaks"):
        audit_source_split(Dataset.from_list([make_row(1)]), Dataset.from_list([make_row(1)]))


def test_invalid_pairs_filter_or_error():
    valid = make_row(1)
    invalid = make_row(2)
    invalid["x0_graph_json"] = json.dumps(TARGET)
    dataset = Dataset.from_list([valid, invalid])
    filtered, audit = audit_and_filter_pairs(dataset, split_name="train", invalid_pair_policy="filter")
    assert len(filtered) == 1
    assert audit.rows_before == 2
    with pytest.raises(ValueError, match="invalid graph pair"):
        audit_and_filter_pairs(dataset, split_name="train", invalid_pair_policy="error")


class MockTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return messages[0]["content"] + "\nMODEL:"


def test_gemma_thinking_is_forwarded_by_default():
    tokenizer = MockTokenizer()
    prompt = apply_graph_completion_chat_template(
        tokenizer,
        render_graph_canvas(EMPTY),
        condition="Humidity increases water uptake.",
    )
    assert tokenizer.kwargs["enable_thinking"] is True
    assert "<answer>" in prompt and "complete corrected graph" in prompt
    assert "Condition:\nHumidity increases water uptake." in prompt


def test_new_trainer_defaults_do_not_change_legacy_defaults():
    args = build_parser().parse_args([])
    assert args.base_model == "google/gemma-4-E4B-it"
    assert args.chat_template_enable_thinking == "true"
    assert args.lora_target_modules == "language-default"
    assert not args.no_lora
    assert args.dtype == "bfloat16"
    assert args.attn_implementation == "paged|sdpa"
    assert args.use_hub_kernels
    assert args.use_transformers_continuous_batching
    assert args.transformers_kv_cache_memory_percent == 0.45
    assert args.transformers_compile_level == 1
    assert not args.transformers_cuda_graphs
    assert args.continuous_batching_unsupported_policy == "fallback"


def test_model_load_defaults_use_bf16_paged_sdpa_and_hub_kernels():
    kwargs = model_load_kwargs()
    assert kwargs["dtype"] is torch.bfloat16
    assert kwargs["attn_implementation"] == "paged|sdpa"
    assert kwargs["use_kernels"] is True


def test_missing_hub_kernels_fails_before_model_download(monkeypatch):
    monkeypatch.setattr(graph_completion_modeling, "is_kernels_available", lambda: False)
    with pytest.raises(ImportError, match="Hub kernels are enabled"):
        load_graph_completion_model_and_tokenizer("does-not-download")


class FakeRolloutModel:
    def __init__(self, config):
        self.config = config
        self.attention = None

    def set_attn_implementation(self, attention):
        self.attention = attention


def test_gemma4_continuous_batching_falls_back_to_sdpa():
    model = FakeRolloutModel(
        SimpleNamespace(
            model_type="gemma4",
            text_config=SimpleNamespace(model_type="gemma4_text"),
        )
    )
    compatible, reason = continuous_batching_compatibility(model)
    assert not compatible
    assert "mixed local/global" in reason
    assert not resolve_continuous_batching(model, requested=True)
    assert model.attention == "sdpa"


def test_supported_decoder_keeps_continuous_batching():
    model = FakeRolloutModel(
        SimpleNamespace(
            model_type="llama",
            num_hidden_layers=4,
            num_attention_heads=4,
            hidden_size=256,
            vocab_size=1024,
        )
    )
    assert resolve_continuous_batching(model, requested=True)
    assert model.attention is None


def test_disabling_continuous_batching_uses_standard_sdpa():
    model = FakeRolloutModel(SimpleNamespace(model_type="llama"))
    assert not resolve_continuous_batching(model, requested=False)
    assert model.attention == "sdpa"


def test_gemma4_continuous_batching_can_be_strict():
    model = FakeRolloutModel(SimpleNamespace(model_type="gemma4", text_config=None))
    with pytest.raises(ValueError, match="does not yet support Gemma 4"):
        resolve_continuous_batching(model, requested=True, unsupported_policy="error")


def test_installed_trl_accepts_sampling_and_kl_settings():
    config = build_compatible_grpo_config(
        output_dir="/tmp/graph-completion-test",
        use_cpu=True,
        bf16=False,
        fp16=False,
        top_p=0.91,
        beta=0.02,
        use_transformers_continuous_batching=True,
        transformers_continuous_batching_config={
            "max_memory_percent": 0.45,
            "use_cuda_graph": False,
            "default_compile_level": 1,
        },
        unsupported_future_option="ignored",
    )
    assert config.top_p == 0.91
    assert config.beta == 0.02
    assert config.use_transformers_continuous_batching
    assert config.transformers_continuous_batching_config["max_memory_percent"] == 0.45
    assert config.transformers_continuous_batching_config["use_cuda_graph"] is False
    assert config.transformers_continuous_batching_config["default_compile_level"] == 1


def test_end_to_end_local_dataset_preparation(tmp_path):
    dataset_path = tmp_path / "dataset"
    DatasetDict(
        {
            "train": Dataset.from_list([make_row(index) for index in range(8)]),
            "test": Dataset.from_list([make_row(index) for index in range(20, 23)]),
        }
    ).save_to_disk(str(dataset_path))
    prepared = prepare_graph_completion_datasets(
        str(dataset_path),
        validation_manifest=str(tmp_path / "manifest.json"),
        validation_source_count=2,
        max_train_rows=3,
        max_eval_rows=2,
    )
    assert len(prepared.train) == 3
    assert len(prepared.validation) == 2
    assert len(prepared.test) == 2
    assert set(prepared.train["source_index"]).isdisjoint(
        set(prepared.validation["source_index"])
    )
    assert "no_edit_primary" in prepared.train.column_names
