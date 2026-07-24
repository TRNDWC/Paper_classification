"""Build multi-label-preserving class subsets for k = 10, 15, 20, 25 (and beyond).

Motivation: restricting the label space by picking the top-N most FREQUENT
primary topics (the original small-scale experiment) accidentally turns the
multi-label problem into a near single-label one (~85% single-label at k=10).
This script instead grows a *densest co-occurrence cluster* of classes
(greedy densest-subgraph heuristic), which preserves far more multi-label
structure. Growth is nested: the k=10 set is a subset of the k=15 set, which
is a subset of k=20, etc. -- so larger-k experiments are natural extensions
of smaller ones, not unrelated re-selections.

For each k in --k_values, this script:
  1. Selects the k classes (densest-cluster growth order).
  2. Samples up to --instances_per_class papers per class (by primary topic_id).
  3. Computes the label cardinality (# positive labels within the k-class
     space) for every sampled paper.
  4. Logs the cardinality frequency table (so_luong_nhan -> so_luong_mau) to
     diagnostics/label_cardinality_k{k}.csv
  5. Saves the selected class list to diagnostics/selected_classes_k{k}.json
     (topic_id + topic name + primary count) so a training script can reuse
     the exact same class set.

A combined summary (avg cardinality, %>=2, %>=3, %>=4 per k) is also saved to
diagnostics/label_cardinality_summary.csv for direct comparison across k.
"""
import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

from dataset import build_label_mapping, load_papers, row_to_labels


def greedy_densest_growth_order(full_labels: np.ndarray, max_k: int) -> List[int]:
    """Return column indices (label2id space) in greedy densest-subgraph growth order."""
    cooc = full_labels.T @ full_labels
    np.fill_diagonal(cooc, 0)

    degree = cooc.sum(axis=1)
    order = [int(np.argmax(degree))]
    remaining = set(range(full_labels.shape[1])) - set(order)

    while len(order) < max_k and remaining:
        gains = {node: cooc[node, order].sum() for node in remaining}
        best_node = max(gains, key=gains.get)
        order.append(best_node)
        remaining.discard(best_node)
    return order


def build_subset(
    papers_df: pd.DataFrame,
    topic_ids: List[int],
    instances_per_class: int,
    seed: int,
) -> pd.DataFrame:
    parts = []
    for t in topic_ids:
        group = papers_df[papers_df["topic_id"] == t]
        n = min(instances_per_class, len(group))
        if n < instances_per_class:
            print(f"  [warn] topic {t} only has {len(group)} primary instances (< {instances_per_class})")
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested densest-cluster class subsets for multiple k.")
    parser.add_argument("--papers_csv", default="papers.csv")
    parser.add_argument("--taxonomy_csv", default="taxonomy.csv")
    parser.add_argument("--k_values", type=int, nargs="+", default=[10, 15, 20, 25])
    parser.add_argument("--instances_per_class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="diagnostics")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    papers_df = load_papers(args.papers_csv)
    label2id, id2label = build_label_mapping(papers_df, args.taxonomy_csv)
    topic_name = papers_df[["topic_id", "topic"]].drop_duplicates().set_index("topic_id")["topic"].to_dict()
    primary_counts = papers_df["topic_id"].value_counts()

    full = np.stack([row_to_labels(row, label2id) for _, row in papers_df.iterrows()])

    max_k = max(args.k_values)
    growth_order = greedy_densest_growth_order(full, max_k)
    growth_topic_ids = [id2label[i] for i in growth_order]

    summary_rows = []
    for k in sorted(args.k_values):
        selected_topics = growth_topic_ids[:k]
        print(f"\n=== k={k} ===")
        for t in selected_topics:
            print(f"  {t}: {topic_name.get(t, '?')} (n_primary={primary_counts.get(t, 0)})")

        subset_df = build_subset(papers_df, selected_topics, args.instances_per_class, args.seed)

        restricted_map: Dict[int, int] = {t: idx for idx, t in enumerate(sorted(selected_topics))}
        restricted = np.stack([row_to_labels(row, restricted_map) for _, row in subset_df.iterrows()])
        cardinality = restricted.sum(axis=1).astype(int)

        freq = Counter(cardinality.tolist())
        freq_df = pd.DataFrame(
            sorted(freq.items()), columns=["so_luong_nhan", "so_luong_mau"]
        )
        freq_df["ty_le_%"] = (100 * freq_df["so_luong_mau"] / len(subset_df)).round(2)

        freq_path = os.path.join(args.out_dir, f"label_cardinality_k{k}.csv")
        freq_df.to_csv(freq_path, index=False)
        print(f"Tổng mẫu: {len(subset_df)} | Tần suất số nhãn/mẫu:")
        print(freq_df.to_string(index=False))
        print(f"-> Đã lưu {freq_path}")

        classes_path = os.path.join(args.out_dir, f"selected_classes_k{k}.json")
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"topic_id": int(t), "topic": topic_name.get(t, "?"), "n_primary": int(primary_counts.get(t, 0))}
                    for t in selected_topics
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"-> Đã lưu danh sách lớp {classes_path}")

        summary_rows.append(
            {
                "k": k,
                "n_papers": len(subset_df),
                "avg_card": round(float(cardinality.mean()), 3),
                "pct_ge2": round(100 * float((cardinality >= 2).mean()), 2),
                "pct_ge3": round(100 * float((cardinality >= 3).mean()), 2),
                "pct_ge4": round(100 * float((cardinality >= 4).mean()), 2),
                "n_ge3": int((cardinality >= 3).sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "label_cardinality_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n=== Tổng hợp so sánh giữa các k ===")
    print(summary_df.to_string(index=False))
    print(f"\nĐã lưu bảng tổng hợp: {summary_path}")


if __name__ == "__main__":
    main()
