#!/bin/bash 

CUDA_VISIBLE_DEVICES=0 python3 baseline_multilabel.py --experiment_name SciBERT_AsymmetricLoss_25K_bs64_P4_N1 \
--model_name_or_path 'allenai/scibert_scivocab_uncased' \
--output_dir '../results' \
--seed 42 \
--eval_strategy steps \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 25000 \
--logging_strategy steps \
--logging_steps 0.05 \
--save_steps 0.2 \
--eval_steps 0.2 \
--Tp 4.0 \
--Tn 1.0 \
--criterion 'AsymmetricLoss' &

CUDA_VISIBLE_DEVICES=1 python3 baseline_multilabel.py --experiment_name SciBERT_AsymmetricLoss_25K_bs64_P1_N1 \
--model_name_or_path 'allenai/scibert_scivocab_uncased' \
--output_dir '../results' \
--seed 42 \
--eval_strategy steps \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 25000 \
--logging_strategy steps \
--logging_steps 0.05 \
--save_steps 0.2 \
--eval_steps 0.2 \
--Tp 1.0 \
--Tn 1.0 \
--criterion 'AsymmetricLoss' &

CUDA_VISIBLE_DEVICES=2 python3 baseline_multilabel.py --experiment_name SciBERT_AsymmetricLoss_25K_bs64_P4_N4 \
--model_name_or_path 'allenai/scibert_scivocab_uncased' \
--output_dir '../results' \
--seed 42 \
--eval_strategy steps \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 25000 \
--logging_strategy steps \
--logging_steps 0.05 \
--save_steps 0.2 \
--eval_steps 0.2 \
--Tp 4.0 \
--Tn 4.0 \
--criterion 'AsymmetricLoss' &

# CUDA_VISIBLE_DEVICES=3 python3 baseline_multilabel.py --experiment_name SciBERT_AsymmetricLoss_25K_bs64_P3_N2 \
# --model_name_or_path 'allenai/scibert_scivocab_uncased' \
# --output_dir '../results' \
# --seed 42 \
# --eval_strategy steps \
# --per_device_train_batch_size 64 \
# --gradient_accumulation_steps 1 \
# --learning_rate 2e-5 \
# --num_train_epochs 1 \
# --max_steps 25000 \
# --logging_strategy steps \
# --logging_steps 0.05 \
# --save_steps 0.2 \
# --eval_steps 0.2 \
# --Tp 3.0 \
# --Tn 2.0 \
# --criterion 'AsymmetricLoss' &

# CUDA_VISIBLE_DEVICES=4 python3 baseline_multilabel.py --experiment_name SciBERT_AsymmetricLoss_25K_bs64_P10_N5 \
# --model_name_or_path 'allenai/scibert_scivocab_uncased' \
# --output_dir '../results' \
# --seed 42 \
# --eval_strategy steps \
# --per_device_train_batch_size 64 \
# --gradient_accumulation_steps 1 \
# --learning_rate 2e-5 \
# --num_train_epochs 1 \
# --max_steps 25000 \
# --logging_strategy steps \
# --logging_steps 0.05 \
# --save_steps 0.2 \
# --eval_steps 0.2 \
# --Tp 10.0 \
# --Tn 5.0 \
# --criterion 'AsymmetricLoss' &


'''
## server - higher batch size
CUDA_VISIBLE_DEVICES=3 python3 baseline_multilabel.py --experiment_name SciBERT_25K_steps_bs64 \
--model_name_or_path 'allenai/scibert_scivocab_uncased' \
--output_dir '../results' \
--seed 42 \
--eval_strategy steps \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--max_steps 25000 \
--logging_strategy steps \
--logging_steps 0.05 \
--save_steps 0.2 \
--eval_steps 0.2 
'''

#0.05*max_steps \ #20 implicit epochs 
#--metric_for_best_model  \
#--label_smoothing_factor 0 \
#--eval_accumulation_steps  \
#baseline_BERT_10K_steps \