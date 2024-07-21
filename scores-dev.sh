#!/bin/bash
source ../venv/bin/activate

eval_files="$(find runs/ -path "*seed-3/eval_results_mlqa-dev" -a -not -path "*mrr*seed-3/eval_results_mlqa-dev" -a -not -path "*klpw-fw*seed-3/eval_results_mlqa-dev"  -a -not -path "*klpw_plus*seed-3/eval_results_mlqa-dev")"
# eval_files="$(find runs/ -path "*seed-3/eval_results_mlqa-dev")"
all_files="$eval_files ./runs/mbert-qa-en/eval_results_mlqa-dev"
echo Found $(echo "$eval_files" | wc -l) evaluation files
python src/evaluation/scores.py "$all_files" f1 mlqa-dev 3 results-all
python src/evaluation/scores.py "$all_files" em mlqa-dev 3 results-all

# find best scores
best_model_file=./runs/best-models-f1-mlqa-dev-3.csv 
echo "model,GXLT,XLT,en-mean,es-mean,de-mean,ar-mean,vi-mean,hi-mean,zh-mean" > $best_model_file
for model in skd_map/  skd/  ce/ kl-fw-map-coeff-self-distil/  kl-fw-self-distil/; do
  best_model=$(grep -P "$model" ./runs/results-all-f1-mlqa-dev-3.csv  | cut -d , -f1,2 | sort  -t, -k 2n,2 | tail -n1 | cut -d, -f1 )
  best_scores=$(grep $best_model ./runs/results-all-f1-mlqa-dev-3.csv)

  echo $best_scores >> $best_model_file
done
