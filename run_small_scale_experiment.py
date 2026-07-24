"""Diagnostic script: small-scale controlled experiment (giao su's item 2).

Two ways to select the k classes:
  --classes_json diagnostics/selected_classes_k{K}.json
      Load the exact class list produced by build_multilabel_subsets.py
      (densest co-occurrence cluster -> preserves multi-label structure).
  (no --classes_json)
      Fall back to the original design: top --num_classes most frequent
      primary topics (this collapses to near single-label, kept only for
      backward comparison).

For the selected classes, samples up to --instances_per_class papers per
class (by primary topic_id, same seeded sampling as build_multilabel_subsets.py
so the resulting subset is identical to the one already logged there), splits
80/20 the same way split_by_primary_topic does, then trains/evaluates the
exact same SciBERTMultiLabelClassifier architecture and hyperparameters as
the full-dataset run (see logs/20260717_144010.txt) so results are directly
comparable across k.
"""
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import (
    PapersDataset,
    load_papers,
    row_to_labels,
    set_seed,
    split_by_primary_topic,
)
from model import (
    SciBERTMultiLabelClassifier,
    compute_pos_weight,
    evaluate,
    run_one_epoch,
)


def build_subset_label_mapping(topic_ids) -> Tuple[Dict[int, int], Dict[int, int]]:
    label_ids = sorted(int(t) for t in topic_ids)
    label2id = {label: idx for idx, label in enumerate(label_ids)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def load_topics_from_json(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return [int(e["topic_id"]) for e in entries]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers_csv", default="papers.csv")
    parser.add_argument("--classes_json", default=None, help="Path to selected_classes_k{k}.json (densest-cluster selection)")
    parser.add_argument("--num_classes", type=int, default=10, help="Used only when --classes_json is not given")
    parser.add_argument("--instances_per_class", type=int, default=100)
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_pos_weight", type=float, default=10.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", default="diagnostics")
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ---- 1. Select classes ----
    papers_df = load_papers(args.papers_csv)
    topic_name = papers_df[["topic_id", "topic"]].drop_duplicates().set_index("topic_id")["topic"].to_dict()
    primary_counts = papers_df["topic_id"].value_counts()

    if args.classes_json:
        selected_topics = load_topics_from_json(args.classes_json)
        print(f"Loaded {len(selected_topics)} classes from {args.classes_json} (densest-cluster selection):")
    else:
        selected_topics = primary_counts.head(args.num_classes).index.tolist()
        print(f"Selected {len(selected_topics)} classes (most frequent primary topic_id):")
    for t in selected_topics:
        print(f"  {t}: {topic_name.get(t, '?')} (n_primary={primary_counts.get(t, 0)})")

    # ---- 2. Sample up to `instances_per_class` papers per selected primary topic ----
    parts = []
    for t in selected_topics:
        group = papers_df[papers_df["topic_id"] == t]
        n = min(args.instances_per_class, len(group))
        if n < args.instances_per_class:
            print(f"  [warn] topic {t} only has {len(group)} instances (< {args.instances_per_class})")
        sampled = group.sample(n=n, random_state=args.seed)
        parts.append(sampled)
    subset_df = pd.concat(parts, ignore_index=True)
    print(f"\nSubset size: {len(subset_df)} papers across {len(selected_topics)} classes")

    # ---- 3. Restrict label space to the selected topics, drop out-of-scope secondary labels ----
    label2id, id2label = build_subset_label_mapping(selected_topics)
    num_labels = len(label2id)

    # ---- 4. Train/test split, same methodology as split_by_primary_topic ----
    train_df, test_df = split_by_primary_topic(subset_df, test_size=args.test_size, seed=args.seed)
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["labels"] = [row_to_labels(row, label2id) for _, row in train_df.iterrows()]
    test_df["labels"] = [row_to_labels(row, label2id) for _, row in test_df.iterrows()]

    train_card = np.stack(train_df["labels"].to_numpy()).sum(axis=1)
    print(f"Avg labels/paper in subset train: {train_card.mean():.2f} | %>=2: {100*(train_card>=2).mean():.1f}% "
          f"| %>=3: {100*(train_card>=3).mean():.1f}%")

    # ---- 5. Tokenizer / DataLoader (same as main pipeline: title only) ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = PapersDataset(train_df, tokenizer=tokenizer, max_length=args.max_length)
    test_dataset = PapersDataset(test_df, tokenizer=tokenizer, max_length=args.max_length)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---- 6. Model / loss / optimizer (mirrors model.py main()) ----
    model = SciBERTMultiLabelClassifier(
        model_name=args.model_name, num_labels=num_labels, dropout=args.dropout
    ).to(device)

    pos_weight = compute_pos_weight(train_loader, device=device, max_weight=args.max_pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(
        "Using BCEWithLogitsLoss with pos_weight "
        f"(min={pos_weight.min().item():.2f}, max={pos_weight.max().item():.2f})"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    os.makedirs(args.log_dir, exist_ok=True)
    if args.log_file is None:
        started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"small_scale_k{num_labels}_{started_at}.txt"
    log_path = os.path.join(args.log_dir, args.log_file)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# k={num_labels} classes, instances_per_class<={args.instances_per_class}, topics={selected_topics}\n")
        f.write("epoch\ttrain_loss\ttest_loss\tmAP\tF1-score\n")
    print(f"Metrics will be saved to {log_path}")

    best_map = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} (k={num_labels}) ==========")
        train_loss = run_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            log_steps=20,
        )
        test_loss, metrics = evaluate(
            model=model, data_loader=test_loader, criterion=criterion, device=device, epoch=epoch
        )
        print(
            f"Epoch {epoch} result | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} "
            f"| mAP: {metrics['mAP']:.4f} | F1-score: {metrics['f1']:.4f}"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t{test_loss:.4f}\t{metrics['mAP']:.4f}\t{metrics['f1']:.4f}\n")
        best_map = max(best_map, metrics["mAP"])

    print(f"\nBest mAP on subset (k={num_labels}): {best_map:.4f}")


if __name__ == "__main__":
    main()
