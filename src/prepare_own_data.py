#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert the papers_{train,test}.csv export into the DatasetDict that the training scripts expect.

The CSVs carry one primary label (`subfield_id`) plus up to four `other_subfield_id_*` columns; those
are folded into a single `cats` list of subfield names, which is the multi-label target format
`baseline_multilabel.py` and `baseline_svm.py` both read.

    python3 prepare_own_data.py --data_dir ../data --output_dir ../data/papers_prepped
"""

import os
import argparse
from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

OTHER_LABEL_COLUMNS = [f"other_subfield_id_{i}" for i in range(1, 5)]
KEEP_COLUMNS = ["id", "title", "abstract", "keyword", "cats", "primary", "field", "strlabel"]


def build_rows(df, id2subfield, id2field):
    """Fold the primary + `other_subfield_id_*` columns into one `cats` list per paper."""
    rows = []
    for i, record in enumerate(df.to_dict("records")):
        label_ids = [int(record["subfield_id"])]
        for column in OTHER_LABEL_COLUMNS:
            value = record.get(column)
            if pd.notna(value):
                label_ids.append(int(value))

        # dict.fromkeys keeps the primary label first while dropping repeats
        label_ids = list(dict.fromkeys(label_ids))
        cats = [id2subfield[label_id] for label_id in label_ids]
        rows.append(
            {
                "id": str(record.get("id", i)),
                "title": str(record["title"]).strip(),
                "abstract": str(record["abstract"]).strip(),
                "keyword": "" if pd.isna(record.get("keyword")) else str(record["keyword"]).strip(),
                "cats": cats,
                "primary": cats[0],
                "field": id2field[label_ids[0]],
                "strlabel": ";".join(sorted(cats)),
            }
        )
    return pd.DataFrame(rows)[KEEP_COLUMNS]


def stratification_key(strlabels, min_count=10):
    """Stratify on the full label set, bucketing combinations too rare to split into 'longtail'."""
    counts = Counter(strlabels)
    return [label if counts[label] >= min_count else "longtail" for label in strlabels]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--output_dir", default=None, help="defaults to <data_dir>/papers_prepped")
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.0,
        help="fraction of train held out as a 'validation' split; 0 (default) keeps all of train "
        "for training, since baseline_multilabel.py evaluates directly against the test split each "
        "epoch and does not use a validation set",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep_leakage",
        action="store_true",
        help="keep duplicate titles and train rows whose title also appears in test (dropped by default)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(args.data_dir, "papers_prepped")

    taxonomy = pd.read_csv(os.path.join(args.data_dir, "taxonomy.csv"))
    id2subfield = dict(zip(taxonomy["subfield_id"], taxonomy["subfield_name"]))
    id2field = dict(zip(taxonomy["subfield_id"], taxonomy["field_name"]))

    train = build_rows(pd.read_csv(os.path.join(args.data_dir, "papers_train.csv")), id2subfield, id2field)
    test = build_rows(pd.read_csv(os.path.join(args.data_dir, "papers_test.csv")), id2subfield, id2field)
    print(f"Loaded train={len(train)} test={len(test)} | {len(id2subfield)} subfields")

    if not args.keep_leakage:
        # only ever drop from train: the test set has to stay exactly as delivered
        before = len(train)
        train = train[~train["title"].str.lower().str.strip().duplicated()]
        deduped = before - len(train)

        test_titles = set(test["title"].str.lower().str.strip())
        train = train[~train["title"].str.lower().str.strip().isin(test_titles)]
        print(f"Dropped {deduped} duplicate titles and {before - deduped - len(train)} train/test overlaps")

    splits = {"train": Dataset.from_pandas(train, preserve_index=False)}
    if args.validation_size > 0:
        train_split, validation_split = train_test_split(
            train,
            test_size=args.validation_size,
            random_state=args.seed,
            stratify=stratification_key(train["strlabel"].tolist()),
        )
        splits["train"] = Dataset.from_pandas(train_split, preserve_index=False)
        splits["validation"] = Dataset.from_pandas(validation_split, preserve_index=False)
    splits["test"] = Dataset.from_pandas(test, preserve_index=False)

    dataset = DatasetDict(splits)
    dataset.save_to_disk(output_dir)

    print(f"\nSaved to {output_dir}")
    for name, split in dataset.items():
        cardinality = sum(len(c) for c in split["cats"]) / len(split)
        print(f"  {name:11s} n={len(split):6d}  label cardinality={cardinality:.2f}")
    print(f"\nColumns: {dataset['train'].column_names}")


if __name__ == "__main__":
    main()
