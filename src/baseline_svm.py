#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TF-IDF + linear SVM baseline for the multi-label subfield classification.

Deliberately shares `build_text` and `multilabel_metrics` with the transformer scripts so the numbers
line up cell for cell. A LinearSVC predicts the positive class when its decision function is > 0, which
is exactly the `sigmoid(logit) > 0.5` rule `multilabel_metrics` applies, so decision values can be fed
in where the transformer passes logits.

    python3 baseline_svm.py --dataset_name ../data/papers_prepped --text_column title
    python3 baseline_svm.py --dataset_name ../data/papers_prepped --compare
"""

import json
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from myutils import build_text, load_any_dataset, multilabel_metrics, seed_everything

METRIC_ORDER = ["f1", "f1_macro", "precision", "recall", "subset_accuracy", "hamming"]


def texts_for(split, text_column):
    return [build_text(example, text_column) for example in split]


def run_one(dataset, text_column, args):
    binarizer = MultiLabelBinarizer()
    y_train = binarizer.fit_transform(dataset["train"][args.label_column])

    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, args.max_ngram),
            min_df=args.min_df,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
        ),
        OneVsRestClassifier(LinearSVC(C=args.C, class_weight="balanced"), n_jobs=-1),
    )
    model.fit(texts_for(dataset["train"], text_column), y_train)

    results = {}
    for split_name in ("validation", "test"):
        if split_name not in dataset:
            continue
        y_true = binarizer.transform(dataset[split_name][args.label_column])
        # decision_function > 0 is the SVM's own boundary, matching the sigmoid > 0.5 rule
        scores = model.decision_function(texts_for(dataset[split_name], text_column))
        results[split_name] = multilabel_metrics(scores, y_true)

    n_features = len(model[0].vocabulary_)
    return results, binarizer.classes_, n_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="../data/papers_prepped")
    parser.add_argument("--text_column", default="title")
    parser.add_argument("--label_column", default="cats")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_ngram", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="sweep title / abstract / title+abstract instead of a single --text_column",
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    dataset = load_any_dataset(args.dataset_name)
    columns = ["title", "abstract", "title+abstract"] if args.compare else [args.text_column]

    all_results = {}
    for text_column in columns:
        results, classes, n_features = run_one(dataset, text_column, args)
        all_results[text_column] = results
        print(f"\n=== TF-IDF + LinearSVC | text={text_column} | {len(classes)} classes, {n_features} features ===")
        for split_name, metrics in results.items():
            line = "  ".join(f"{key}={metrics[key]:.4f}" for key in METRIC_ORDER)
            print(f"  [{split_name:10s}] {line}")

    if len(columns) > 1:
        print(f"\n{'text_column':16s} " + " ".join(f"{key:>16s}" for key in METRIC_ORDER))
        for text_column, results in all_results.items():
            metrics = results["test"]
            print(f"{text_column:16s} " + " ".join(f"{metrics[key]:16.4f}" for key in METRIC_ORDER))

    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(all_results, handle, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
