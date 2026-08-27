#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2024 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"

## Necessary installs
#!pip install datasets transformers evaluate sentencepiece accelerate

import os
import sys
import numpy as np
from argparse import Namespace
from transformers import AutoTokenizer, HfArgumentParser, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments
from myutils import (
    CustomArguments,
    seed_everything,
    load_any_dataset,
    preprocess_function,
    MultiLabelTrainingArguments,
    MultiLabelTrainer,
    sigmoid,
    compute_metrics,
)


def main():
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    custom_args, prior_training_args = parser.parse_args_into_dataclasses()
    for k, v in custom_args.__dict__.items():
        print(k, v)
    args = Namespace(**vars(custom_args), **vars(prior_training_args))
    seed_everything(args.seed)

    # TrainingArguments resolves a missing --report_to to every installed integration (e.g. wandb)
    # during argument parsing itself, before this line -- so args.report_to is never None here even
    # when the flag was never passed. That auto-detected wandb then blocks on a login/API-key prompt
    # if the package happens to be present but unconfigured. Require it to be opt-in on the CLI.
    if "--report_to" not in sys.argv:
        args.report_to = "none"

    ## Load the dataset and initialize the classes
    # DATAROOT = os.path.join(os.path.dirname(__file__), "..", "data")
    # dataset = load_from_disk(os.path.join(DATAROOT, "arxiv_dataset_prepped"))

    # # make another subset of validation - still too large
    # simple_validation = dataset["validation"].train_test_split(
    #     test_size=0.1, seed=args.seed, stratify_by_column="strlabel"
    # )
    # dataset["simple_validation"] = simple_validation["test"]

    dataset = load_any_dataset(args.dataset_name)  # hub id, save_to_disk dir, or local csv/json

    label_name = args.label_column

    classes = sorted(set([c for cats in dataset["train"][label_name] for c in cats]))
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}

    print(f"Classes: {len(classes)}")
    print(f"Class2id: {class2id}")
    print(f"Id2class: {id2class}")
    print(f"Input text column(s): {args.text_column} @ max_seq_length {args.max_seq_length}")

    ## Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenized_dataset = dataset.map(
        lambda example: preprocess_function(
            example,
            class2id,
            tokenizer,
            label_name=label_name,
            text_column=args.text_column,
            max_length=args.max_seq_length,
        )
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # no held-out validation set: evaluate directly against test every epoch, as requested. This
    # means metric_for_best_model / load_best_model_at_end pick a checkpoint using test performance,
    # so the final numbers are not a blind evaluation -- fine for iterating quickly, but for a
    # protocol-clean number to report, prep the data with --validation_size > 0 and evaluate on that
    # split instead.
    if "test" not in tokenized_dataset:
        holdout = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        tokenized_dataset["train"], tokenized_dataset["test"] = holdout["train"], holdout["test"]
    eval_split = test_split = "test"
    print(f"Splits: train / eval=test / test=test (n_train={len(tokenized_dataset['train'])}, "
          f"n_test={len(tokenized_dataset['test'])})")
    print("WARNING: no validation split -- checkpoint selection is based on test performance.")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(classes),
        id2label=id2class,
        label2id=class2id,
        problem_type="multi_label_classification",
    )

    training_args = MultiLabelTrainingArguments(
        # new
        criterion=args.criterion,
        Tp=args.Tp,
        Tn=args.Tn,
        # old
        output_dir=os.path.join(args.output_dir, args.experiment_name),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        # "epoch" guarantees one train-loss + eval line per epoch no matter the dataset size or
        # batch size; the original "steps" + fractional eval_steps/save_steps was tuned for the
        # 25K-step arxiv run and produced far sparser, epoch-misaligned logging on a smaller dataset
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_first_step=True,
        learning_rate=args.learning_rate,  # override for now
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        weight_decay=0.01,  # override default
        warmup_ratio=0.1,  # override default
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=3,
        push_to_hub=args.push_to_hub,
        hub_strategy="end",
        load_best_model_at_end=True,
        run_name=args.experiment_name,
        hub_model_id=args.experiment_name,
        label_smoothing_factor=args.label_smoothing_factor,
        # this block rebuilds TrainingArguments field by field, so anything not forwarded here is
        # silently dropped from the command line -- these are the runtime knobs a GPU run needs
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset[eval_split],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        train_results = trainer.train()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
    except KeyboardInterrupt as e:
        print(e)

    test_dataset = tokenized_dataset[test_split]
    subsample_test = test_dataset.select(range(min(10000, len(test_dataset))))  # takes 30 minutes on desktop
    trainer.evaluate(eval_dataset=subsample_test, metric_key_prefix="test")  # 10K samples is enough?

    if args.push_to_hub:
        trainer.push_to_hub(f"Saving best model of {args.experiment_name} to hub")

    # print some example outputs
    subset = subsample_test.select(range(min(100, len(subsample_test))))
    probabilities = sigmoid(trainer.predict(subset).predictions)
    predictions = (probabilities > 0.5).astype(int)
    references = np.array(subset["labels"]).astype(int)

    print("Example outputs to check:")
    for i in range(len(subset)):
        predicted = [f"{id2class[j]}@{probabilities[i][j]:.2f}" for j in np.flatnonzero(predictions[i])]
        gold = [id2class[j] for j in np.flatnonzero(references[i])]
        print(f"P:{predicted} vs. G:{gold}")

if __name__ == "__main__":
    main()
