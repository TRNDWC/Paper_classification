import os
from dataclasses import dataclass, field
import random
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support


@dataclass
class CustomArguments:
    # DEV: would be cleaner to separate per training approach
    experiment_name: str = field(
        metadata={"help": "Name of the experiment"},
    )
    model_name_or_path: str = field(
        default="bert-base-uncased",  # "microsoft/deberta-v3-small", "allenai/scibert_scivocab_uncased"
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    dataset_name: str = field(
        default="jordyvl/arxiv_dataset_prep",
        metadata={
            "help": """Dataset to train/evaluate on. Either a hub identifier, a directory saved with
                  `save_to_disk`, or a .csv/.json/.jsonl file (a single file is loaded as the 'train' split)."""
        },
    )
    text_column: str = field(
        default="abstract",
        metadata={
            "help": """Dataset column(s) used as model input: 'abstract', 'title', or several columns
                  joined with '+' (e.g. 'title+abstract'), which are concatenated with '. '."""
        },
    )
    label_column: str = field(
        default="cats",
        metadata={"help": "Dataset column holding the list of gold labels per example"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": """Truncation length in tokens. 512 suits abstracts; titles are ~15 tokens, so use
                  something like 64 for --text_column title to avoid wasting compute on padding."""
        },
    )
    criterion: str = field(
        default="BCEWithLogitsLoss",
        metadata={"help": "The loss function to use: 'BCEWithLogitsLoss', 'TwoWayLoss'"},
    )
    Tp: float = field(
        default=4.0,
        metadata={"help": "Temperature for positive logits in TwoWayLoss"},
    )
    Tn: float = field(
        default=1.0,
        metadata={"help": "Temperature for negative logits in TwoWayLoss"},
    )
    sentence_transformer: str = field(
        default="sentence-transformers/paraphrase-mpnet-base-v2",  # "jordyvl/scibert_scivocab_uncased_sentence_transformer"
        metadata={"help": "Path to pretrained sentence transformer model"},
    )
    LLM: str = field(
        default="NousResearch/Llama-2-7b-hf",
        metadata={"help": "Path to pretrained sentence transformer model"},
    )
    multi_target_strategy: str = field(
        default="one-vs-rest",
        metadata={
            "help": """The multi-target strategy to use. Possible values are:
                  "one-vs-rest": uses a OneVsRestClassifier head.
                "multi-output": uses a MultiOutputClassifier head.
                "classifier-chain": uses a ClassifierChain head."""
        },
    )
    use_differentiable_head: bool = field(
        default=False,
        metadata={"help": "Whether to use differentiable heads for multi-target models."},
    )


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_any_dataset(dataset_name):
    """Load a hub identifier, a `save_to_disk` directory, or a local csv/json/jsonl file.

    A single local file has no splits of its own and lands in 'train'; the caller carves an
    evaluation split out of it.
    """
    if os.path.isdir(dataset_name):
        return load_from_disk(dataset_name)
    if os.path.isfile(dataset_name):
        extension = os.path.splitext(dataset_name)[1].lstrip(".").lower()
        return load_dataset("json" if extension == "jsonl" else extension, data_files=dataset_name)
    return load_dataset(dataset_name)


def build_text(example, text_column="abstract"):
    """Assemble the model input from one or more dataset columns.

    A '+' joins columns (e.g. 'title+abstract'), skipping the ones that are empty for this example.
    """
    columns = text_column.split("+")
    missing = [c for c in columns if c not in example]
    if missing:
        raise KeyError(
            f"--text_column asked for {missing}, which the dataset does not have. "
            f"Available columns: {sorted(example.keys())}"
        )
    parts = [(example[c] or "").strip() for c in columns]
    return ". ".join(part for part in parts if part)


def preprocess_function(example, class2id, tokenizer, label_name="cats", text_column="abstract", max_length=512):
    text = build_text(example, text_column)
    all_labels = example[label_name]
    labels = np.zeros(len(class2id))
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1.0

    # padded dynamically per batch by DataCollatorWithPadding, so no padding to max_length here
    example = tokenizer(text, truncation=True, max_length=max_length)
    example["labels"] = labels
    return example


def preprocess_zero_function(example, class2id, label_name="strlabel"):
    all_labels = [c.strip() for c in example[label_name].split(";")]
    labels = np.zeros(len(class2id))
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1.0
    example["label"] = labels
    return example


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hamming_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)


# DEV:  port mAP for classes and instances from https://github.com/tk1980/TwoWayMultiLabelLoss/blob/main/utils/utils.py ?


def multilabel_metrics(logits, references, threshold=0.5):
    """Standard multi-label scores over a (n_samples, n_labels) binarized matrix.

    `accuracy`/`precision`/`recall`/`f1` stay micro-averaged over label cells, matching what earlier
    runs logged. `subset_accuracy` is the stricter exact-match ratio, and the macro scores weigh every
    class equally, which is what the long-tailed arxiv label distribution actually stresses.
    """
    predictions = (sigmoid(np.asarray(logits)) > threshold).astype(int)
    references = np.asarray(references).astype(int)

    metrics = {
        "accuracy": float((predictions == references).mean()),  # label-wise, i.e. 1 - hamming
        "subset_accuracy": float((predictions == references).all(axis=1).mean()),
        "hamming": float(hamming_loss(references, predictions)),
    }
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average=average, zero_division=0
        )
        suffix = "" if average == "micro" else "_macro"
        metrics[f"precision{suffix}"] = float(precision)
        metrics[f"recall{suffix}"] = float(recall)
        metrics[f"f1{suffix}"] = float(f1)
    return metrics


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return multilabel_metrics(predictions, labels)


def setfit_compute_metrics(predictions, labels, **metric_kwargs):
    return multilabel_metrics(predictions.numpy(), np.array(labels))


class TwoWayLoss(nn.Module):
    """
    Implementation of the two-way loss function from the paper:

    @inproceedings{kobayashi2023cvpr,
    title={Two-way Multi-Label Loss},
    author={Takumi Kobayashi},
    booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
    }

    Not well described in the paper, yet there is an obvious batch size effect from the sample mask
    """

    def __init__(self, Tp=4.0, Tn=1.0):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp  # temperature for positive logits
        self.Tn = Tn  # temperature for negative logits
        self.nINF = -100

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, self.nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x / self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x / self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]

        nmask = y.masked_fill(y != 0, self.nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x / self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x / self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return (
            torch.nn.functional.softplus(nlogit_class + plogit_class).mean()  # classwise
            + torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()  # samplewise
        )


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class MultiLabelTrainingArguments(TrainingArguments):
    """Extending original TrainingArguments to include advanced hyperparameters"""

    def __init__(self, *args, criterion=None, Tp=4.0, Tn=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion
        self.Tp = Tp
        self.Tn = Tn


class MultiLabelTrainer(Trainer):
    """Extending original Trainer class to include more advanced training strategies"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_criterion()

    def init_criterion(self):
        self.criterion = None
        if self.args.criterion == "TwoWayLoss":
            self.criterion = TwoWayLoss(Tp=self.args.Tp, Tn=self.args.Tn)
        if self.args.criterion == "AsymmetricLoss":
            self.criterion = AsymmetricLoss(
                gamma_neg=self.args.Tp, gamma_pos=self.args.Tn
            )  # reusing args for gamma_neg and gamma_pos

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Overriding compute_loss to include alternative loss functions when training

        `num_items_in_batch` is passed by transformers >= 4.46 and is unused here: both custom
        criteria already reduce over the batch themselves.
        """
        outputs = model(**inputs)
        loss = outputs.loss  # default for multilabel is BCEWithLogitsLoss
        if self.criterion is not None:
            loss = self.criterion(outputs.logits, inputs["labels"])
        return (loss, outputs) if return_outputs else loss
