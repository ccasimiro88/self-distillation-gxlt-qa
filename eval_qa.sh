#!/bin/bash
script_dir="$(cd "$(dirname "$0")" && pwd)"

source $script_dir/venv/bin/activate

model_dir=$(realpath $1)
testset=$2

echo Evaluate model: $model_dir on testset: $testset
model_type=bert
seed=1

if [[ $testset == mlqa-test ]]; then
    langs="en es de ar hi vi zh"
    data_dir=$script_dir/corpora/MLQA_V1/test
elif [[ $testset == mlqa-dev ]]; then
    langs="en es de ar hi vi zh"
    data_dir=$script_dir/corpora/MLQA_V1/dev
elif [[ $testset == tydiqa-goldp ]]; then
    langs="english arabic bengali finnish indonesian korean russian swahili telugu"
    data_dir=$script_dir/corpora/tydiqa-goldp-v1.1-dev
elif [[ $testset == xquad ]]; then
    langs="en es de ar hi vi zh el ru tr th"
    data_dir=$script_dir/corpora/xquad
fi

cache_eval_dir=$script_dir/runs/.cache_eval/mbert-qa-en
eval_dir=$model_dir/preds
mkdir -p $eval_dir
mkdir -p $cache_eval_dir

if [[ ! -f $model_dir/eval_results_$testset && -f $model_dir/pytorch_model.bin ]]; then
    # Compute predictions
    predict_files=$(find $data_dir -name "*.json")
    python $script_dir/src/run_squad_w_distillation.py \
        --model_type $model_type  \
        --model_name_or_path $model_dir \
        --do_eval \
        --predict_files $predict_files \
        --per_gpu_eval_batch_size 8 \
        --max_seq_length 384 \
        --threads 1 \
        --seed $seed  \
        --eval_dir $eval_dir \
        --cache_eval_dir $cache_eval_dir \
        --output_dir $model_dir

    # Compute language-specific metrics
    if [[ $testset == mlqa-test ]]; then
        for cl in $langs; do
        for ql in $langs; do
            predict_file=$data_dir/test-context-$cl-question-$ql.json
            prediction_file=$eval_dir/predictions_test-context-$cl-question-$ql.json
        
            python $script_dir/src/metrics/mlqa_evaluation_v1.py \
                --dataset_file $predict_file \
                --dataset_type $testset \
                --prediction_file $prediction_file \
                --results_file $model_dir/eval_results_$testset \
                --context_language $cl \
                --question_language $ql 
        done 
        done

    elif [[ $testset == mlqa-dev ]]; then
        for cl in $langs; do
        for ql in $langs; do
            predict_file=$data_dir/dev-context-$cl-question-$ql.json
            prediction_file=$eval_dir/predictions_dev-context-$cl-question-$ql.json


            python $script_dir/src/metrics/mlqa_evaluation_v1.py \
                --dataset_file $predict_file \
                --dataset_type $testset \
                --prediction_file $prediction_file \
                --results_file $model_dir/eval_results_$testset \
                --context_language $cl \
                --question_language $ql 

        done
        done

    elif [[ $testset == tydiqa-goldp ]]; then
        for l in $langs; do
            predict_file=$data_dir/tydiqa-goldp-dev-$l.json
            prediction_file=$eval_dir/predictions_tydiqa-goldp-dev-$l.json

            python $script_dir/src/metrics/evaluate_v1.py \
                --dataset_file $predict_file \
                --dataset_type $testset \
                --prediction_file $prediction_file \
                --results_file $model_dir/eval_results_$testset \
                --context_language $l \
                --question_language $l 
        done

    elif [[ $testset == xquad ]]; then
        for l in $langs; do
            predict_file=$data_dir/xquad.$l.json
            prediction_file=$eval_dir/predictions_xquad.$l.json

            python $script_dir/src/metrics/evaluate_v1.py \
                --dataset_file $predict_file \
                --dataset_type $testset \
                --prediction_file $prediction_file \
                --results_file $model_dir/eval_results_$testset \
                --context_language $l \
                --question_language $l 
        done
    fi
fi

