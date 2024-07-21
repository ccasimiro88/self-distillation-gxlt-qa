#!/bin/bash
source ../venv/bin/activate

for metric in f1 em; do
# mBERT-qa-en
eval_file="./runs/mbert-qa-en/eval_results_mlqa-dev"
suffix="mBERT-qa-en, zero-shot"
testset=mlqa-dev
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset --metric $metric
eval_file="./runs/mbert-qa-en/eval_results_mlqa-test"
suffix="mBERT-qa-en, zero-shot"
testset=mlqa-test
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset --metric $metric

# skd + map
eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd_map/temp-2/seed-3/eval_results_mlqa-test
suffix="mBERT-qa-en, skd + MAP@k"
testset=mlqa-test
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric
eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd_map/temp-2/seed-3/eval_results_mlqa-dev
suffix="mBERT-qa-en, skd + MAP@k"
testset=mlqa-dev
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric
eval_files=$(find ./runs/joint_train-xquad -path "*skd_map*eval_results_mlqa-dev")
python src/figures/heatmap.py --files $eval_files --suffix "$suffix" --testset $testset --heatmap_type "temp-vs-ntl"  --metric $metric

# skd 
eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd/temp-2/seed-3/eval_results_mlqa-test
suffix="mBERT-qa-en, skd"
testset=mlqa-test
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric
eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd/temp-2/seed-3/eval_results_mlqa-dev
suffix="mBERT-qa-en, skd"
testset=mlqa-dev
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset
eval_files=$(find ./runs/joint_train-xquad -path "*skd*eval_results_mlqa-dev")
python src/figures/heatmap.py --files $eval_files --suffix "$suffix" --testset $testset --heatmap_type "temp-vs-ntl" --metric $metric

# CE 
eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3/eval_results_mlqa-test
suffix="mBERT-qa-en, ce"
testset=mlqa-test
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric
eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3/eval_results_mlqa-dev
suffix="mBERT-qa-en, ce"
testset=mlqa-dev
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric
done