#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ..
pwdPath="$(pwd)"

#python -m src.train_pte \
#  --epochs 200 \
#  --batch_size 32 \
#  --log_freq 100 \
#  --task_name 'SG' \
#  --model_path /home/han/llh/EvoShield/bert-base-uncased \
#  --train_data_path /home/han/llh/EvoShield/data/safe-guard-prompt-injection/full_train_16.csv \
#  --test_data_path /home/han/llh/EvoShield/data/safe-guard-prompt-injection/full_test.csv
#
#python -m src.train_pte \
#  --epochs 200 \
#  --batch_size 32 \
#  --log_freq 100 \
#  --task_name 'PI' \
#  --model_path /home/han/llh/EvoShield/bert-base-uncased \
#  --train_data_path /home/han/llh/EvoShield/data/prompt-injections/full_train_16.csv \
#  --test_data_path /home/han/llh/EvoShield/data/prompt-injections/full_test.csv

python -m src.train_pte \
  --epochs 200 \
  --batch_size 32 \
  --log_freq 100 \
  --task_name 'JC' \
  --model_path /home/han/llh/EvoShield/bert-base-uncased \
  --train_data_path /home/han/llh/EvoShield/data/jailbreak-classification-imbalanced/full_train_16.csv \
  --test_data_path /home/han/llh/EvoShield/data/jailbreak-classification-imbalanced/full_test.csv

python -m src.train_pte \
  --epochs 200 \
  --batch_size 32 \
  --log_freq 100 \
  --task_name 'JC' \
  --model_path /home/han/llh/EvoShield/bert-base-uncased \
  --train_data_path /home/han/llh/EvoShield/data/jailbreak-classification-balanced/full_train_16.csv \
  --test_data_path /home/han/llh/EvoShield/data/jailbreak-classification-balanced/full_test.csv
