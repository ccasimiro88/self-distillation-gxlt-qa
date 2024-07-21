#!/bin/bash
testset=$1

best_checkpoints="runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd_map/temp-2/seed-3
                  runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/skd/temp-2/seed-3
                  runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-3/ce/seed-3
                  runs/mbert-qa-en"


ablation="runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/kl-fw-map-coeff-self-distil/temp-2/seed-3
          runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-5/kl-fw-self-distil/temp-2/seed-3"


for model in $best_checkpoints; do 
    bash eval_qa.sh $(realpath $model) $testset
done

for model in $ablation; do 
    bash eval_qa.sh $(realpath $model) $testset
done

# srun -p veu --mem=10G --gres=gpu:1 --pty bash eval-qa-test.sh mlqa-test
# srun -p veu --mem=10G --gres=gpu:1 --pty bash eval-qa-test.sh tydiqa-goldp
# srun -p veu --mem=10G --gres=gpu:1 --pty bash eval-qa-test.sh xquad