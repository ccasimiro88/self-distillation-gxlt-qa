#!/bin/bash
# Training script for standard QA training with Cross-Entropy loss and different types of distillation, Kullback-Leibler divergence and MSE loss 
script_dir="$(cd "$(dirname "$0")" && pwd)"

source $script_dir/venv/bin/activate

# Constants
kl_type="fw"
seed=3
losses="kl"
train_langs="en es de ar vi hi zh"
epochs=3
model=$script_dir/runs/mbert-qa-en
model_type=bert
train_data_types=xquad
train_data_dirs=$script_dir/corpora/xquad
# select only the best config on MLQA-dev for this ablation experiment
ntl="5"
temps="2"


for ntl in $ntl; do
for temp in $temps; do
  out_dir=$script_dir/runs/joint_train-$train_data_types/${train_langs// /-}/mbert-qa-en/ep-$epochs/ntl-$ntl/${losses// /-}-$kl_type-map-coeff-self-distil/temp-$temp/seed-$seed
  mkdir -p $out_dir

  if [[ ! -f $out_dir/pytorch_model.bin ]]; then

      python $script_dir/src/sampler_qa.py \
             --train-data-dirs  $train_data_dirs \
             --train-data-types $train_data_types \
             --languages $train_langs \
             --num-tgt-langs $ntl \
             --seed $seed \
             --output-dir $out_dir

      logging_dir=$out_dir/tb
      mkdir -p $logging_dir

    python $script_dir/src/run_squad_w_distillation.py \
                                --model_type $model_type  \
                                --model_name_or_path $model \
                                --teacher_type $model_type \
                                --teacher_name_or_path $model \
                                --losses $losses \
                                --use_map_loss_coefficients \
                                --self_distil \
                                --alpha_ce 0 \
                                --kl_type $kl_type \
                                --temperature $temp \
                                --do_train \
                                --train_file $out_dir/train.json \
                                --per_gpu_train_batch_size 6 \
                                --gradient_accumulation_steps 4 \
                                --max_seq_length 384 \
                                --num_train_epochs $epochs \
                                --logging_steps 50 \
                                --save_steps 10000000 \
                                --overwrite_output_dir \
                                --threads 1 \
                                --seed $seed  \
                                --logging_dir $logging_dir \
                                --output_dir $out_dir

  fi

  if [[ ! -f $out_dir/eval_results_$testset ]]; then
    bash $script_dir/eval_qa.sh $out_dir mlqa-dev
  fi
done
done
