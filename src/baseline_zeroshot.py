#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2024 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"

#!pip3 install sentence_transformers setfit --upgrade

import os
import numpy as np
import random
from argparse import Namespace
from datasets import load_dataset
from transformers import HfArgumentParser
from transformers import TrainingArguments as HFTrainingArguments
try:
    import wandb
except ImportError:  # optional: runs fine without experiment tracking installed
    wandb = None
import evaluate
from setfit import SetFitModel, Trainer, TrainingArguments
from myutils import (
    CustomArguments,
    seed_everything,
    preprocess_zero_function,
    MultiLabelTrainingArguments,
    MultiLabelTrainer,
    sigmoid,
    setfit_compute_metrics,
)


def main():
    parser = HfArgumentParser((CustomArguments, HFTrainingArguments))
    custom_args, prior_training_args = parser.parse_args_into_dataclasses()
    for k, v in custom_args.__dict__.items():
        print(k, v)
    args = Namespace(**vars(custom_args), **vars(prior_training_args))
    seed_everything(args.seed)

    if wandb is not None:
        wandb.init(
            project="scientific-text-classification",
            name=args.experiment_name if args.experiment_name else None,
            tags=["zero-shot"],
            config={k: v for k, v in args.__dict__.items() if v is not None},
        )
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.experiment_name),
        max_steps=args.max_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        batch_size=args.per_device_train_batch_size,
        save_total_limit=3,
        load_best_model_at_end=True,
        run_name=args.experiment_name,
        num_iterations=1,  # just once
    )

    ## Load the dataset and initialize the classes; https://github.com/huggingface/setfit/issues/226
    dataset = load_dataset("jordyvl/arxiv_dataset_prep")
    classes = sorted(set([c.strip() for cats in dataset["train"]["strlabel"] for c in cats.split(";")]))
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}

    print(f"Classes: {len(classes)}")

    # for setfit compatibility need 'label' (onehot) and 'text' columns
    dataset = dataset.map(lambda x: preprocess_zero_function(x, class2id), remove_columns=["strlabel"]).rename_column(
        "abstract", "text"
    )

    # do subsampling here already to avoid memory issues
    ##

    # DEV: https://github.com/huggingface/setfit/issues/472 cannot use whole dataset due to memory issues; raised ticket
    ## numpy.core._exceptions._ArrayMemoryError: Unable to allocate 3.72 TiB for an array with shape (2022879, 2022879) and data type bool
    ## related to https://stackoverflow.com/questions/40617199/generate-n-choose-2-combinations-in-python-on-very-large-data-sets
    ## backoff solution: subsample training data to % of the original size

    # sample a random 10K entries
    samples = random.sample(range(len(dataset["train"])), 10000)
    validation_samples = random.sample(range(len(dataset["simple_validation"])), 2000)
    test_samples = list(range(0, 10000))
    # samples = validation_samples = test_samples = list(range(10))  # debugging
    dataset["train"] = dataset["train"].select(samples)
    dataset["simple_validation"] = dataset["simple_validation"].select(validation_samples)
    subsample_test = dataset["test"].select(test_samples)

    model = SetFitModel.from_pretrained(
        args.sentence_transformer,
        multi_target_strategy=args.multi_target_strategy,
        use_differentiable_head=args.use_differentiable_head,
        labels=classes,
    )

    model = model.to("cuda")

    # Create trainer
    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["simple_validation"],
        metric=setfit_compute_metrics,  # "f1",  # DEV: no evaluate.compute metrics availability, TODO: custom setfit_compute_metrics and evaluation loop
        # metric_kwargs={"average": "micro"},
    )
    try:
        train_results = trainer.train()
    except KeyboardInterrupt as e:
        print(e)

    res = trainer.evaluate(subsample_test, metric_key_prefix="test")  # 10K samples is enough?
    print(res)
    if wandb is not None:
        wandb.log({f"test/{k}": v for k, v in res.items()})

    # print some example outputs
    trainer.push_to_hub(f"jordyvl/{args.experiment_name}")
    print("Example outputs to check:")
    subset = subsample_test.select(list(range(0, 100)))

    preds = sigmoid(trainer.predict(subset))
    predictions = (preds > 0.5).astype(int).reshape(-1)
    references = np.array(subset["labels"]).astype(int).reshape(-1)

    # convert to classes
    for i in range(len(subset)):
        print(
            f"P:{id2class[predictions[i]]} @{preds:{round(preds[i],2)}} vs. G:{id2class[references[i]]}"
        )  # f'P:{predictions[i]} vs. G:{references[i]}'


if __name__ == "__main__":
    main()
