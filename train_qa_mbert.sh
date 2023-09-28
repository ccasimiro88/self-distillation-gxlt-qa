#!/bin/bash
script_dir="$(cd "$(dirname "$0")" && pwd)"

source $script_dir/venv/bin/activate

# Constants
model=bert-base-multilingual-cased
testsets="mlqa-test mlqa-dev"
train_file=squad
testset="squad-dev"
train_method="ce"

out_dir=$script_dir/runs/mbert-qa-en
logging_dir=$out_dir/tb    
mkdir -p $logging_dir
mkdir -p $out_dir

python $script_dir/src/run_qa.py \
  --model_name_or_path $model \
  --dataset_name $train_file \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --save_strategy no \
  --logging_steps 50 \
  --logging_strategy steps \
  --logging_dir $logging_dir \
  --output_dir $out_dir