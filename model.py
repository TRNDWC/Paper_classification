import argparse
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
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


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = min(k, y_score.shape[1])
    top_k_indices = np.argsort(-y_score, axis=1)[:, :k]
    scores = []
    for row_index, indices in enumerate(top_k_indices):
        scores.append(float(y_true[row_index, indices].sum()) / k)
    return float(np.mean(scores))


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    k: int,
) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int32)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mean_ap = average_precision_score(y_true, y_score, average="macro")
    p_at_k = precision_at_k(y_true, y_score, k=k)
    return {
        "macro_f1": float(macro_f1),
        "mAP": float(mean_ap),
        f"precision@{k}": float(p_at_k),
        "threshold": float(threshold),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.05,
) -> Tuple[float, float]:
    best_threshold = min_threshold
    best_f1 = -1.0

    thresholds = np.arange(min_threshold, max_threshold + 1e-8, step)
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(np.int32)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if macro_f1 > best_f1:
            best_f1 = float(macro_f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1


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
    threshold: float,
    k: int,
    epoch: int,
    tune_threshold: bool,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
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
    if tune_threshold:
        threshold, _ = find_best_threshold(
            y_true,
            y_score,
            min_threshold=threshold_min,
            max_threshold=threshold_max,
            step=threshold_step,
        )
    metrics = compute_metrics(y_true, y_score, threshold=threshold, k=k)
    return total_loss / max(1, len(data_loader)), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SciBERT for multi-label topic classification.")
    parser.add_argument("--train_csv_path", default="papers_train.csv")
    parser.add_argument("--test_csv_path", default="papers_test.csv")
    parser.add_argument("--taxonomy_path", default="taxonomy.csv")
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune_threshold", action="store_true", default=True)
    parser.add_argument("--no_tune_threshold", dest="tune_threshold", action="store_false")
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--precision_k", type=int, default=5)
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

    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_file is None:
        started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"{started_at}.txt"
    log_path = os.path.join(args.output_dir, args.log_file)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            "epoch\ttrain_loss\ttest_loss\tthreshold\tMacro-F1\tmAP\t"
            f"Precision@{args.precision_k}\n"
        )
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
            threshold=args.threshold,
            k=args.precision_k,
            epoch=epoch,
            tune_threshold=args.tune_threshold,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )

        print(
            f"Epoch {epoch} result | train_loss: {train_loss:.4f} "
            f"| test_loss: {test_loss:.4f} "
            f"| threshold: {metrics['threshold']:.2f} "
            f"| Macro-F1: {metrics['macro_f1']:.4f} "
            f"| mAP: {metrics['mAP']:.4f} "
            f"| Precision@{args.precision_k}: {metrics[f'precision@{args.precision_k}']:.4f}"
        )
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"{epoch}\t"
                f"{train_loss:.6f}\t"
                f"{test_loss:.6f}\t"
                f"{metrics['threshold']:.6f}\t"
                f"{metrics['macro_f1']:.6f}\t"
                f"{metrics['mAP']:.6f}\t"
                f"{metrics[f'precision@{args.precision_k}']:.6f}\n"
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
