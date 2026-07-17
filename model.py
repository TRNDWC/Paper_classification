import argparse
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from dataset import create_dataloaders, set_seed


class SciBERTMultiLabelClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.scibert = AutoModel.from_pretrained(model_name)
        hidden_size = self.scibert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.scibert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_vector = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_vector))
        return logits


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    mean_ap = average_precision_score(y_true, y_score, average="macro")
    epsilon = 1e-8
    true_positive = (y_score * y_true).sum()
    false_positive = (y_score * (1.0 - y_true)).sum()
    false_negative = ((1.0 - y_score) * y_true).sum()
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    f1 = 2.0 * precision * recall / (precision + recall + epsilon)
    return {
        "mAP": float(mean_ap),
        "f1": float(f1),
    }


def compute_pos_weight(data_loader, device: torch.device, max_weight: float) -> torch.Tensor:
    labels = np.stack(data_loader.dataset.df["labels"].to_numpy())
    positive_counts = labels.sum(axis=0)
    negative_counts = labels.shape[0] - positive_counts
    pos_weight = negative_counts / np.maximum(positive_counts, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, max_weight)
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def run_one_epoch(
    model: nn.Module,
    data_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    log_steps: int,
) -> float:
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch} training",
        leave=True,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(progress_bar, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")

        optimizer.zero_grad(set_to_none=True)
        logits = model(**batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / step
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")
        # if step == 1 or step % log_steps == 0 or step == len(data_loader):
        #     print(
        #         f"Epoch {epoch} | train step {step}/{len(data_loader)} "
        #         f"| loss: {loss.item():.4f}"
        #     )

    return total_loss / max(1, len(data_loader))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    criterion,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch} testing",
        leave=True,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(progress_bar, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")

        logits = model(**batch)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        avg_loss = total_loss / step
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        # if step == 1 or step == len(data_loader):
        #     print(
        #         f"Epoch {epoch} | test step {step}/{len(data_loader)} "
        #         f"| loss: {loss.item():.4f}"
        #     )

    y_true = np.vstack(all_labels)
    y_score = np.vstack(all_probs)
    metrics = compute_metrics(y_true, y_score)
    return total_loss / max(1, len(data_loader)), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SciBERT for multi-label topic classification.")
    parser.add_argument("--train_csv_path", default="papers_train.csv")
    parser.add_argument("--test_csv_path", default="papers_test.csv")
    parser.add_argument("--taxonomy_path", default="taxonomy.csv")
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_pos_weight", action="store_true", default=True)
    parser.add_argument("--no_pos_weight", dest="use_pos_weight", action="store_false")
    parser.add_argument("--max_pos_weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, test_loader, label2id, id2label = create_dataloaders(
        train_csv_path=args.train_csv_path,
        test_csv_path=args.test_csv_path,
        tokenizer=tokenizer,
        taxonomy_path=args.taxonomy_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    num_labels = len(label2id)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of labels: {num_labels}")

    model = SciBERTMultiLabelClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)

    if args.use_pos_weight:
        pos_weight = compute_pos_weight(
            train_loader,
            device=device,
            max_weight=args.max_pos_weight,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(
            "Using BCEWithLogitsLoss with pos_weight "
            f"(min={pos_weight.min().item():.2f}, max={pos_weight.max().item():.2f})"
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    if args.log_file is None:
        started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"{started_at}.txt"
    log_path = os.path.join(args.log_dir, args.log_file)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("epoch\ttrain_loss\ttest_loss\tmAP\tF1-score\n")
    print(f"Metrics will be saved to {log_path}")

    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")
        train_loss = run_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            log_steps=args.log_steps,
        )
        test_loss, metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        print(
            f"Epoch {epoch} result | train_loss: {train_loss:.4f} "
            f"| test_loss: {test_loss:.4f} "
            f"| mAP: {metrics['mAP']:.4f} "
            f"| F1-score: {metrics['f1']:.4f}"
        )
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"{epoch}\t"
                f"{train_loss:.4f}\t"
                f"{test_loss:.4f}\t"
                f"{metrics['mAP']:.4f}\t"
                f"{metrics['f1']:.4f}\n"
            )

        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )
            print(f"Saved best model to {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
