#!/bin/bash
source ./venv-new/bin/activate

for metric in F1 EM; do
# mBERT-qa-en
eval_file="./runs/mbert-qa-en/eval_results_mlqa-dev"
suffix="mBERT-qa-en, ZS, MLQA-dev"
testset=MLQA-dev
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset --metric $metric

eval_file="./runs/mbert-qa-en/eval_results_mlqa-test"
suffix="mBERT-qa-en, ZS, MLQA-test"
testset=MLQA-test
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset --metric $metric

# skd + map
eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/ce-kl-fw-map-coeff-self-distil/temp-2/seed-3/eval_results_mlqa-test
suffix="mBERT-qa-en, CLS + SKD, mAP@10"
testset=MLQA-test
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric

eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/ce-kl-fw-map-coeff-self-distil/temp-2/seed-3/eval_results_mlqa-dev
suffix="mBERT-qa-en, CLS + SKD, mAP@10"
testset=MLQA-dev
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric

eval_files=$(find ./runs/joint_train-xquad -path "*ce-kl-fw-map-coeff-self-distil*eval_results_mlqa-dev")
python src/figures/heatmap.py --files $eval_files --suffix "$suffix" --testset $testset --heatmap_type "temp-vs-ntl"  --metric $metric

# skd 
eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/ce-kl-fw-self-distil/temp-2/seed-3/eval_results_mlqa-test
suffix="mBERT-qa-en, CLS + SKD"
testset=MLQA-test
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric

eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/ce-kl-fw-self-distil/temp-2/seed-3/eval_results_mlqa-dev
suffix="mBERT-qa-en, CLS + SKD"
testset=MLQA-dev
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset --metric $metric

eval_files=$(find ./runs/joint_train-xquad -path "*ce-kl-fw-self-distil*eval_results_mlqa-dev")
python src/figures/heatmap.py --files $eval_files --suffix "$suffix" --testset $testset --heatmap_type "temp-vs-ntl" --metric $metric

# CE 
eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3/eval_results_mlqa-test
suffix="mBERT-qa-en, CLS + CE"
testset=MLQA-test
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric

eval_file=./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3/eval_results_mlqa-dev
suffix="mBERT-qa-en, CLS + CE"
testset=MLQA-dev
python src/figures/heatmap.py --file $eval_file --suffix "$suffix" --testset $testset  --metric $metric
done
