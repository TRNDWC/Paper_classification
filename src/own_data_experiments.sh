#!/bin/bash
# Experiments on the in-house papers_{train,test}.csv export (30 Scopus subfields, multi-label).
# Run from inside src/ :  bash own_data_experiments.sh
#
# Reference to beat, TF-IDF + LinearSVC on the same splits (test set, 5947 docs):
#   title           F1_micro 0.5089  F1_macro 0.5133  subset_acc 0.1557
#   abstract        F1_micro 0.5641  F1_macro 0.5678  subset_acc 0.2405
#   title+abstract  F1_micro 0.5846  F1_macro 0.5913  subset_acc 0.2563
#
# Logging runs once per epoch (train loss + full eval metrics), not per step -- baseline_multilabel.py
# hardcodes eval/save/logging_strategy to "epoch".
#
# If backgrounding this (nohup, screen, &), capture BOTH streams: transformers' logger and its tqdm
# progress bar both write to stderr by default, so `bash own_data_experiments.sh > train.log &` alone
# silently drops every epoch line. Use:
#   nohup bash own_data_experiments.sh > train.log 2>&1 &
#   tail -f train.log

set -euo pipefail

DATA='../data/papers_prepped'
MODEL='bert-base-uncased'   # then: allenai/scibert_scivocab_uncased, microsoft/deberta-v3-small

# fp16 for T4/V100, bf16 for A100/A6000/RTX 30xx and newer. Set exactly one to True.
PRECISION='--fp16 True'
# PRECISION='--bf16 True'

# '--report_to none' keeps the run offline; swap to 'wandb' after `pip install wandb && wandb login`
TRACKING='--report_to none'

# checkpoint selection on F1 rather than the default eval_loss: with 30 imbalanced labels the loss
# keeps improving after F1 has peaked
BEST='--metric_for_best_model eval_f1 --greater_is_better True'

COMMON="--output_dir ../results --seed 42 \
--learning_rate 2e-5 --num_train_epochs 5 --lr_scheduler_type cosine \
--dataloader_num_workers 4 $PRECISION $TRACKING $BEST"

# --- title only ---------------------------------------------------------------
# titles average 11.5 words (p99 = 22), so 64 subword tokens covers essentially all of them
python3 baseline_multilabel.py --experiment_name BERT_own_title \
--model_name_or_path "$MODEL" --dataset_name "$DATA" \
--text_column 'title' --max_seq_length 64 \
--per_device_train_batch_size 64 --gradient_accumulation_steps 1 \
--criterion 'BCEWithLogitsLoss' $COMMON

# --- abstract only ------------------------------------------------------------
# abstracts average 160 words / p95 = 271, so 384 tokens keeps ~95% of them intact
python3 baseline_multilabel.py --experiment_name BERT_own_abstract \
--model_name_or_path "$MODEL" --dataset_name "$DATA" \
--text_column 'abstract' --max_seq_length 384 \
--per_device_train_batch_size 16 --gradient_accumulation_steps 2 \
--criterion 'BCEWithLogitsLoss' $COMMON

# --- title + abstract (best SVM setting, so the fair comparison point) ---------
python3 baseline_multilabel.py --experiment_name BERT_own_title_abstract \
--model_name_or_path "$MODEL" --dataset_name "$DATA" \
--text_column 'title+abstract' --max_seq_length 384 \
--per_device_train_batch_size 16 --gradient_accumulation_steps 2 \
--criterion 'BCEWithLogitsLoss' $COMMON

# --- class-imbalance-aware loss ------------------------------------------------
# 21K training rows over 30 labels is small, and the rarest class has 624 examples against 4475 for
# the most common. Try this before concluding that more data is needed.
python3 baseline_multilabel.py --experiment_name BERT_own_title_abstract_ASL \
--model_name_or_path "$MODEL" --dataset_name "$DATA" \
--text_column 'title+abstract' --max_seq_length 384 \
--per_device_train_batch_size 16 --gradient_accumulation_steps 2 \
--criterion 'AsymmetricLoss' --Tp 4.0 --Tn 1.0 $COMMON
