#!/bin/bash
script_dir="$(cd "$(dirname "$0")" && pwd)"

# CE
model_dir=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3
model="ce"

# MLQA
mlqa_langs="en es de ar hi vi zh"

# MLQA-dev
testset=mlqa-dev
results_file=$model_dir/eval_results_example_level_${testset}_$model

if [ ! -f "$results_file" ]; then
    for cl in $mlqa_langs; do
        for ql in $mlqa_langs; do
        
        predict_file=$script_dir/corpora/MLQA_V1/dev/dev-context-$cl-question-$ql.json
        prediction_file=$model_dir/preds/predictions_dev-context-$cl-question-$ql.json


        python $script_dir/src/metrics/example_level_scores.py \
                        --dataset_file $predict_file \
                        --dataset_type $testset \
                        --prediction_file $prediction_file \
                        --model $model \
                        --results_file $model_dir/eval_results_example_level_${testset}_$model \
                        --context_language $cl \
                        --question_language $ql 
        done
    done
fi

# MLQA-test
testset=mlqa-test
results_file=$model_dir/eval_results_example_level_${testset}_$model

if [ ! -f "$results_file" ]; then
    for cl in $mlqa_langs; do
        for ql in $mlqa_langs; do
       
        predict_file=$script_dir/corpora/MLQA_V1/test/test-context-$cl-question-$ql.json
        prediction_file=$model_dir/preds/predictions_test-context-$cl-question-$ql.json


        python $script_dir/src/metrics/example_level_scores.py \
                        --dataset_file $predict_file \
                        --dataset_type $testset \
                        --prediction_file $prediction_file \
                        --model $model \
                        --results_file $model_dir/eval_results_example_level_${testset}_$model \
                        --context_language $cl \
                        --question_language $ql 
        done
    done
fi

# XQUAD
xquad_langs="en es de ar hi vi zh el ru tr th"
testset=xquad
results_file=$model_dir/eval_results_example_level_${testset}_$model
if [ ! -f "$results_file" ]; then
    for l in $xquad_langs; do
        predict_file=$script_dir/corpora/xquad/xquad.$l.json
        prediction_file=$model_dir/preds/predictions_xquad.$l.json

        python $script_dir/src/metrics/example_level_scores.py \
                    --dataset_file $predict_file \
                    --dataset_type $testset \
                    --prediction_file $prediction_file \
                    --model $model \
                    --results_file $model_dir/eval_results_example_level_${testset}_$model \
                    --context_language $l \
                    --question_language $l 
    done
fi

# TyDiQA-goldp
tydiqa_langs="english arabic bengali finnish indonesian korean russian swahili telugu"
testset=tydiqa-goldp
results_file=$model_dir/eval_results_example_level_${testset}_$model
if [ ! -f "$results_file" ]; then
    for l in $tydiqa_langs; do
        predict_file=$script_dir/corpora/tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-$l.json
        prediction_file=$model_dir/preds/predictions_tydiqa-goldp-dev-$l.json

        python $script_dir/src/metrics/example_level_scores.py \
                        --dataset_file $predict_file \
                        --dataset_type $testset \
                        --prediction_file $prediction_file \
                        --model $model \
                        --results_file $model_dir/eval_results_example_level_${testset}_$model \
                        --context_language $l \
                        --question_language $l 
    done
fi

# SKD
model_dir=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd/temp-2/seed-3/seed-3
model="skd"

# MLQA
mlqa_langs="en es de ar hi vi zh"

# MLQA-dev
testset=mlqa-dev
results_file=$model_dir/eval_results_example_level_${testset}_$model

if [ ! -f "$results_file" ]; then
    for cl in $mlqa_langs; do
        for ql in $mlqa_langs; do
        
        predict_file=$script_dir/corpora/MLQA_V1/dev/dev-context-$cl-question-$ql.json
        prediction_file=$model_dir/preds/predictions_dev-context-$cl-question-$ql.json


        python $script_dir/src/metrics/example_level_scores.py \
                        --dataset_file $predict_file \
                        --dataset_type $testset \
                        --prediction_file $prediction_file \
                        --model $model \
                        --results_file $model_dir/eval_results_example_level_${testset}_$model \
                        --context_language $cl \
                        --question_language $ql 
        done
    done
fi

# MLQA-test
testset=mlqa-test
results_file=$model_dir/eval_results_example_level_${testset}_$model

if [ ! -f "$results_file" ]; then
    for cl in $mlqa_langs; do
        for ql in $mlqa_langs; do
       
        predict_file=$script_dir/corpora/MLQA_V1/test/test-context-$cl-question-$ql.json
        prediction_file=$model_dir/preds/predictions_test-context-$cl-question-$ql.json


        python $script_dir/src/metrics/example_level_scores.py \
                        --dataset_file $predict_file \
                        --dataset_type $testset \
                        --prediction_file $prediction_file \
                        --model $model \
                        --results_file $model_dir/eval_results_example_level_${testset}_$model \
                        --context_language $cl \
                        --question_language $ql 
        done
    done
fi

# XQUAD
xquad_langs="en es de ar hi vi zh el ru tr th"
testset=xquad
results_file=$model_dir/eval_results_example_level_${testset}_$model
if [ ! -f "$results_file" ]; then
    for l in $xquad_langs; do
        predict_file=$script_dir/corpora/xquad/xquad.$l.json
        prediction_file=$model_dir/preds/predictions_xquad.$l.json

        python $script_dir/src/metrics/example_level_scores.py \
                    --dataset_file $predict_file \
                    --dataset_type $testset \
                    --prediction_file $prediction_file \
                    --model $model \
                    --results_file $model_dir/eval_results_example_level_${testset}_$model \
                    --context_language $l \
                    --question_language $l 
    done
fi

# TyDiQA-goldp
tydiqa_langs="english arabic bengali finnish indonesian korean russian swahili telugu"
testset=tydiqa-goldp
results_file=$model_dir/eval_results_example_level_${testset}_$model
if [ ! -f "$results_file" ]; then
    for l in $tydiqa_langs; do
        predict_file=$script_dir/corpora/tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-$l.json
        prediction_file=$model_dir/preds/predictions_tydiqa-goldp-dev-$l.json

        python $script_dir/src/metrics/example_level_scores.py \
                        --dataset_file $predict_file \
                        --dataset_type $testset \
                        --prediction_file $prediction_file \
                        --model $model \
                        --results_file $model_dir/eval_results_example_level_${testset}_$model \
                        --context_language $l \
                        --question_language $l 
    done
fi


# SKD-MAP
model_dir=$script_dir/runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd_map/temp-2/seed-3/seed-3
model="skd_map"

# MLQA
mlqa_langs="en es de ar hi vi zh"

# MLQA-dev
testset=mlqa-dev
results_file=$model_dir/eval_results_example_level_${testset}_$model

if [ ! -f "$results_file" ]; then
    for cl in $mlqa_langs; do
        for ql in $mlqa_langs; do
        
        predict_file=$script_dir/corpora/MLQA_V1/dev/dev-context-$cl-question-$ql.json
        prediction_file=$model_dir/preds/predictions_dev-context-$cl-question-$ql.json


        python $script_dir/src/metrics/example_level_scores.py \
                        --dataset_file $predict_file \
                        --dataset_type $testset \
                        --prediction_file $prediction_file \
                        --model $model \
                        --results_file $model_dir/eval_results_example_level_${testset}_$model \
                        --context_language $cl \
                        --question_language $ql 
        done
    done
fi

# MLQA-test
testset=mlqa-test
results_file=$model_dir/eval_results_example_level_${testset}_$model

if [ ! -f "$results_file" ]; then
    for cl in $mlqa_langs; do
        for ql in $mlqa_langs; do
       
        predict_file=$script_dir/corpora/MLQA_V1/test/test-context-$cl-question-$ql.json
        prediction_file=$model_dir/preds/predictions_test-context-$cl-question-$ql.json


        python $script_dir/src/metrics/example_level_scores.py \
                        --dataset_file $predict_file \
                        --dataset_type $testset \
                        --prediction_file $prediction_file \
                        --model $model \
                        --results_file $model_dir/eval_results_example_level_${testset}_$model \
                        --context_language $cl \
                        --question_language $ql 
        done
    done
fi

# XQUAD
xquad_langs="en es de ar hi vi zh el ru tr th"
testset=xquad
results_file=$model_dir/eval_results_example_level_${testset}_$model
if [ ! -f "$results_file" ]; then
    for l in $xquad_langs; do
        predict_file=$script_dir/corpora/xquad/xquad.$l.json
        prediction_file=$model_dir/preds/predictions_xquad.$l.json

        python $script_dir/src/metrics/example_level_scores.py \
                    --dataset_file $predict_file \
                    --dataset_type $testset \
                    --prediction_file $prediction_file \
                    --model $model \
                    --results_file $model_dir/eval_results_example_level_${testset}_$model \
                    --context_language $l \
                    --question_language $l 
    done
fi

# TyDiQA-goldp
tydiqa_langs="english arabic bengali finnish indonesian korean russian swahili telugu"
testset=tydiqa-goldp
results_file=$model_dir/eval_results_example_level_${testset}_$model
if [ ! -f "$results_file" ]; then
    for l in $tydiqa_langs; do
        predict_file=$script_dir/corpora/tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-$l.json
        prediction_file=$model_dir/preds/predictions_tydiqa-goldp-dev-$l.json

        python $script_dir/src/metrics/example_level_scores.py \
                        --dataset_file $predict_file \
                        --dataset_type $testset \
                        --prediction_file $prediction_file \
                        --model $model \
                        --results_file $model_dir/eval_results_example_level_${testset}_$model \
                        --context_language $l \
                        --question_language $l 
    done
fi