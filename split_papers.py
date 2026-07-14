import argparse

from dataset import load_papers, split_by_primary_topic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split papers.csv into train/test CSV files by primary topic_id."
    )
    parser.add_argument("--input_csv", default="papers.csv")
    parser.add_argument("--train_csv", default="papers_train.csv")
    parser.add_argument("--test_csv", default="papers_test.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    papers_df = load_papers(args.input_csv)
    train_df, test_df = split_by_primary_topic(
        papers_df,
        test_size=args.test_size,
        seed=args.seed,
    )

    train_df.to_csv(args.train_csv, index=False)
    test_df.to_csv(args.test_csv, index=False)

    print(f"Saved train file: {args.train_csv}")
    print(f"Saved test file: {args.test_csv}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train primary topics: {train_df['topic_id'].nunique()}")
    print(f"Test primary topics: {test_df['topic_id'].nunique()}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
