#!/bin/bash
# baseline_multilabel.py hardcodes eval/save/logging_strategy to "epoch" -- no --logging_steps /
# --save_steps / --eval_steps to pass here, they'd be silently ignored.
python3 baseline_multilabel.py --experiment_name test_AsymmetricLoss_25K_bs64_P4_N1 \
--model_name_or_path 'allenai/scibert_scivocab_uncased' \
--output_dir '../results' \
--seed 42 \
--per_device_train_batch_size 8  \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 10 \
--Tp 4.0 \
--Tn 1.0 \
--criterion 'AsymmetricLoss'\

"""

CUDA_VISIBLE_DEVICES=3 python3 baseline_multilabel.py --experiment_name test_twowayloss_implementation \
--model_name_or_path 'bert-base-uncased' \
--output_dir '../results' \
--seed 42 \
--eval_strategy steps \
--per_device_train_batch_size 8  \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 10 \
--logging_strategy steps \
--logging_steps 0.5 \
--save_steps 0.5 \
--eval_steps 0.5 \
--Tp 4.0 \
--Tn 1.0 \
--criterion 'TwoWayLoss' \

CUDA_VISIBLE_DEVICES=2 python3 baseline_generative.py --experiment_name Llama2_test \
--LLM 'NousResearch/Llama-2-7b-hf' \
--output_dir '../results' \
--seed 42 \
--eval_strategy steps \
--per_device_train_batch_size 8  \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 10 \
--logging_strategy steps \
--logging_steps 0.5 \
--save_steps 0.5 \
--eval_steps 0.5 \

CUDA_VISIBLE_DEVICES=2 python3 baseline_zeroshot.py --experiment_name test_implementation \
--sentence_transformer 'jordyvl/scibert_scivocab_uncased_sentence_transformer' \
--output_dir '../results' \
--seed 42 \
--eval_strategy steps \
--per_device_train_batch_size 8  \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 10 \
--logging_strategy steps \
--logging_steps 0.5 \
--save_steps 0.5 \
--eval_steps 0.5 \
"""