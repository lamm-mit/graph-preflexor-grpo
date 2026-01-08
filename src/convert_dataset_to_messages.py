#!/usr/bin/env python3
'''
# Default (prompt/chosen) - private repo
python convert_dataset_to_messages.py \
  --source lamm-mit/graph_reasoning_v3 \
  --target lamm-mit/graph_reasoning_v3_messages

# Custom columns
python convert_dataset_to_messages.py \
  --source lamm-mit/graph_reasoning_v3 \
  --target lamm-mit/graph_reasoning_v3_messages \
  --prompt-col instruction \
  --chosen-col response

# Public repo
python convert_dataset_to_messages.py \
  --source lamm-mit/graph_reasoning_v3 \
  --target lamm-mit/graph_reasoning_v3_messages \
  --hub_public

# Multiple datasets merged (note the quoting)
python convert_dataset_to_messages.py \
  --source "org/dataset_a | org/dataset_b" \
  --target org/combined_messages

# Disable shuffling
python convert_dataset_to_messages.py \
  --source lamm-mit/graph_reasoning_v3 \
  --target lamm-mit/graph_reasoning_v3_messages \
  --no-shuffle

# Custom shuffle seed
python convert_dataset_to_messages.py \
  --source lamm-mit/graph_reasoning_v3 \
  --target lamm-mit/graph_reasoning_v3_messages \
  --seed 123
'''
import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a HF dataset to messages-only format and push to the Hub."
    )
    p.add_argument(
        "--source",
        type=str,
        default="lamm-mit/graph_reasoning_v3",
        help="Source dataset on the Hugging Face Hub (e.g., org/name).",
    )
    p.add_argument(
        "--target",
        type=str,
        default="lamm-mit/graph_reasoning_v3_messages",
        help="Target dataset repo on the Hugging Face Hub (e.g., org/name).",
    )
    p.add_argument(
        "--prompt-col",
        type=str,
        default="prompt",
        help="Column name to use as the user prompt (default: prompt).",
    )
    p.add_argument(
        "--chosen-col",
        type=str,
        default="chosen",
        help="Column name to use as the assistant response (default: chosen).",
    )
    p.add_argument(
        "--hub_public",
        action="store_true",
        help="Make Hub repo public (default: private).",
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling the dataset before pushing (default: shuffle enabled).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42).",
    )
    return p.parse_args()

def main():
    args = parse_args()

    sources = [s.strip() for s in args.source.split("|") if s.strip()]
    if not sources:
        raise ValueError("No valid datasets provided in --source (expected format: 'ds1 | ds2 | ...').")

    def to_messages(example):
        # Provide a clear error if a split doesn't contain expected columns
        if args.prompt_col not in example:
            raise KeyError(f"Missing prompt column '{args.prompt_col}' in example keys: {list(example.keys())}")
        if args.chosen_col not in example:
            raise KeyError(f"Missing chosen column '{args.chosen_col}' in example keys: {list(example.keys())}")

        return {
            "messages": [
                {"role": "user", "content": example[args.prompt_col]},
                {"role": "assistant", "content": example[args.chosen_col]},
            ]
        }

    new_splits = {}
    for source in sources:
        dataset = load_dataset(source)
        for split, ds in dataset.items():
            # Validate columns exist in this split
            missing = [c for c in (args.prompt_col, args.chosen_col) if c not in ds.column_names]
            if missing:
                raise ValueError(
                    f"Dataset '{source}' split '{split}' is missing columns {missing}. Available columns: {ds.column_names}"
                )

            mapped = ds.map(
                to_messages,
                remove_columns=ds.column_names,
            )

            if split in new_splits:
                new_splits[split] = concatenate_datasets([new_splits[split], mapped])
            else:
                new_splits[split] = mapped

    new_dataset = DatasetDict(new_splits)

    # Shuffle the dataset if not disabled
    if not args.no_shuffle:
        print(f"Shuffling dataset with seed {args.seed}...")
        new_dataset = DatasetDict({
            split: ds.shuffle(seed=args.seed)
            for split, ds in new_dataset.items()
        })

    new_dataset.push_to_hub(args.target, private=(not args.hub_public))

if __name__ == "__main__":
    main()
