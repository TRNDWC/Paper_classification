#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2024 Jordy Van Landeghem"
__license__ = "GPL v3"
__version__ = "3.0"

## Necessary installs
#!pip install datasets transformers evaluate sentencepiece accelerate munkres

import os
import numpy as np
from argparse import Namespace
import torch
from datasets import load_dataset, load_from_disk, ClassLabel, Sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    HfArgumentParser,
)
from tqdm import tqdm

from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from transformers import TrainingArguments
try:
    import wandb
except ImportError:  # optional: runs fine without experiment tracking installed
    wandb = None
from myutils import CustomArguments, seed_everything
from munkres import Munkres, make_cost_matrix


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein distance between two strings."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def ANLS(y: list[str], y_hat: list[str], threshold=0.5) -> float:
    """Compute Average Normalized Levenshtein Similarity between each element of the lists at the same index and average it."""

    def NLS(gt: str, pred: str) -> float:
        pred = pred.lower().strip()
        gt = gt.lower().strip()

        dist = levenshtein_distance(gt, pred)
        length = max(len(gt.upper()), len(pred.upper()))
        res = 1 - (0.0 if length == 0 else float(dist) / float(length))
        if res < threshold:
            res = 0
        return res

    return sum([NLS(gt, pred) for gt, pred in zip(y, y_hat)]) / len(y)


def exact_match(y: list[str], y_hat: list[str]) -> float:
    """Compute Exact Match between each element of the lists at the same index and average it."""

    def EM(gt: str, pred: str) -> int:
        pred = pred.lower().strip()
        gt = gt.lower().strip()
        return int(gt == pred)

    return sum([EM(gt, pred) for gt, pred in zip(y, y_hat)]) / len(y)


def get_best_matches_hungarian_munkers(anchor_list, matching_list, threshold=0.5):
    match_dict = {}
    match_matrix = []
    for anchor_item in anchor_list:
        NLS_dict = {}
        NLS_list = []
        for matching_item in matching_list:
            NLS = ANLS([anchor_item], matching_item, threshold=threshold)
            NLS_dict[str(matching_item) + " "] = NLS
            NLS_list.append(NLS)

        match_dict[anchor_item] = NLS_dict
        match_matrix.append(NLS_list)

    return match_dict, match_matrix


def split_unique_original_order(lst):
    splitted = [x.split(";") for x in lst]
    for i, cats in enumerate(splitted):
        seen = []
        for cat in cats:
            if cat not in seen:
                seen.append(cat)
        splitted[i] = seen
    return splitted


def NLSL(gt_list, pred_list, threshold=0.5) -> float:
    if len(gt_list) < len(pred_list):
        anchor_list, matching_list = gt_list, pred_list

    else:
        anchor_list, matching_list = pred_list, gt_list

    match_dict, cost_matrix = get_best_matches_hungarian_munkers(anchor_list, matching_list, threshold=threshold)
    num_answers = max(len(set(gt_list)), len(pred_list))

    m = Munkres()
    m_cost_matrix = make_cost_matrix(cost_matrix)
    indexes = m.compute(m_cost_matrix)
    values = [cost_matrix[row][column] for row, column in indexes]
    NLSL = np.sum(values) / num_answers

    return NLSL


def ANLSL(gt_list, pred_list, threshold=0.5) -> float:
    gt_list = split_unique_original_order(gt_list)
    pred_list = split_unique_original_order(pred_list)
    return np.mean([ANLS(gt, pred, threshold=threshold) for gt, pred in zip(gt_list, pred_list)])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred  # split by " ; " each, then ANLSL
    return


def load_llm(args):
    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(args.LLM, quantization_config=bnb_config, device_map="auto")
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.LLM, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    return tokenizer, model, peft_config


PROMPT_DICT = {
    "prompt_instruction": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{document}\n\n### Response:{response}"
    ),
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{document}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def row_to_prompt(row, prompt_type="prompt_instruction", max_input_tokens=None):
    instruction = "Classify this arxiv research article abstract into one or more of the categories (multi-label), separated by ';'."
    document = row["abstract"]
    gt = response = row["strlabel"]

    prompt = PROMPT_DICT[prompt_type]
    if "instruction" in prompt_type:
        text = prompt.format_map({"instruction": instruction, "document": document, "response": response})
    else:
        text = prompt.format_map({"instruction": instruction, "document": document})
    if max_input_tokens is not None:
        text = text[:max_input_tokens]
    return text, gt


def batched_row_to_prompt(batch, prompt_type="prompt_task", max_input_tokens=None):
    texts = []
    gts = []
    for i in range(len(batch["id"])):
        row = {k: v[i] for k, v in batch.items()}
        text, gt = row_to_prompt(row, prompt_type=prompt_type, max_input_tokens=max_input_tokens)
        texts.append(text)
        gts.append(gt)
    return {"instruction": texts, "gt": gts}


def create_instruction_tuning_data():
    dataset = load_dataset("jordyvl/arxiv_dataset_prep")
    instruction_dataset = dataset.map(
        lambda x: batched_row_to_prompt(x, prompt_type="prompt_instruction", max_input_tokens=2048),
        batched=True,
        batch_size=8,
        num_proc=1,
    )
    # DEV: coding could be improved here, \eg map to remove response line, prototyping fast for now
    instruction_dataset["test"] = dataset["test"].map(
        lambda x: batched_row_to_prompt(x, prompt_type="prompt_input", max_input_tokens=2048),
        batched=True,
        batch_size=8,
        num_proc=1,
    )
    instruction_dataset = instruction_dataset.remove_columns((["abstract", "primary", "secondary"]))
    instruction_dataset.save_to_disk("../data/arxiv_dataset_instructions")


def main():
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    custom_args, prior_training_args = parser.parse_args_into_dataclasses()
    for k, v in custom_args.__dict__.items():
        print(k, v)
    args = Namespace(**vars(custom_args), **vars(prior_training_args))
    seed_everything(args.seed)

    if wandb is not None:
        wandb.init(
            project="scientific-text-classification",
            name=args.experiment_name if args.experiment_name else None,
            tags=["generative"],
            config={k: v for k, v in args.__dict__.items() if v is not None},
        )

    instruction_data = "../data/arxiv_dataset_instructions"
    if not os.path.exists(instruction_data):
        create_instruction_tuning_data()

    ## Load the dataset and initialize the classes
    # create instruction for finetuning
    dataset = load_from_disk(instruction_data)

    tokenizer, model, peft_config = load_llm(args)

    # Set training parameters

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.experiment_name),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=2e-4,  # override for now
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        weight_decay=0.001,  # override default
        warmup_ratio=0.03,  # override default
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=3,
        push_to_hub=True,
        hub_strategy="end",
        load_best_model_at_end=True,
        run_name=args.experiment_name,
        hub_model_id=args.experiment_name,
        label_smoothing_factor=args.label_smoothing_factor,
        group_by_length=True,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        fp16=False,
        bf16=False,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["simple_validation"].select(range(10)),
        peft_config=peft_config,
        dataset_text_field="instruction",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        # DEV: https://github.com/huggingface/trl/issues/862  no option for compute_metrics
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(args.experiment_name)

    # Empty VRAM in case of training
    # del pipe
    del model
    del trainer
    import gc

    gc.collect()
    gc.collect()

    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        args.LLM,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.experiment_name)
    model = model.merge_and_unload()

    # Do evaluation here manually

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    all_pred = []
    all_gt = []
    for example in tqdm(dataset["test"].select(list(range(0, 10)))):
        prompt = example["instruction"].split("Response:")[0] + "Response:"
        pred = pipe(prompt)[0]["generated_text"]  # output entity is max size 100?
        pred = pred.strip()[len(prompt) :]

        all_pred.append(pred)
        all_gt.append(str(example["gt"]))

    # evaluate with exact match or ANLS
    accuracy = exact_match(all_gt, all_pred)
    anls = ANLSL(all_gt, all_pred)
    print(f"Exact match accuracy: {accuracy}")
    print(f"ANLS: {anls}")
    if wandb is not None:
        wandb.log({"test/accuracy": accuracy, "test/anls": anls})


if __name__ == "__main__":
    main()
