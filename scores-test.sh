#!/bin/bash
source ../venv/bin/activate

# F1
eval_files="$(find runs/ -path "*seed-3/eval_results_mlqa-test"  -a -not -path "*mrr*seed-3/eval_results_mlqa-test" -a -not -path "*klpw-fw*seed-3/eval_results_mlqa-test"  -a -not -path "*klpw_plus*seed-3/eval_results_mlqa-test")"
all_files="$eval_files ./runs/mbert-qa-en/eval_results_mlqa-test"
echo Found $(echo "$all_files" | wc -l) evaluation files
python src/evaluation/scores.py "$all_files" f1 mlqa-test 3 results-best

eval_files="$(find runs/ -path "*seed-3/eval_results_xquad"  -a -not -path "*mrr*seed-3/eval_results_mlqa-test" -a -not -path "*klpw-fw*seed-3/eval_results_mlqa-test"  -a -not -path "*klpw_plus*seed-3/eval_results_mlqa-test")"
all_files="$eval_files ./runs/mbert-qa-en/eval_results_xquad"
echo Found $(echo "$all_files" | wc -l) evaluation files
python src/evaluation/scores.py "$all_files" f1 xquad 3 results-best

eval_files="$(find runs/ -path "*seed-3/eval_results_tydiqa-goldp"  -a -not -path "*mrr*seed-3/eval_results_mlqa-test" -a -not -path "*klpw-fw*seed-3/eval_results_mlqa-test"  -a -not -path "*klpw_plus*seed-3/eval_results_mlqa-test")"
all_files="$eval_files ./runs/mbert-qa-en/eval_results_tydiqa-goldp"
echo Found $(echo "$all_files" | wc -l) evaluation files
python src/evaluation/scores.py "$all_files" f1 tydiqa-goldp 3 results-best

# EM
eval_files="$(find runs/ -path "*seed-3/eval_results_mlqa-test"  -a -not -path "*mrr*seed-3/eval_results_mlqa-test" -a -not -path "*klpw-fw*seed-3/eval_results_mlqa-test"  -a -not -path "*klpw_plus*seed-3/eval_results_mlqa-test")"
all_files="$eval_files ./runs/mbert-qa-en/eval_results_mlqa-test"
echo Found $(echo "$all_files" | wc -l) evaluation files
python src/evaluation/scores.py "$all_files" em mlqa-test 3 results-best

eval_files="$(find runs/ -path "*seed-3/eval_results_xquad"  -a -not -path "*mrr*seed-3/eval_results_mlqa-test" -a -not -path "*klpw-fw*seed-3/eval_results_mlqa-test"  -a -not -path "*klpw_plus*seed-3/eval_results_mlqa-test")"
all_files="$eval_files ./runs/mbert-qa-en/eval_results_xquad"
echo Found $(echo "$all_files" | wc -l) evaluation files
python src/evaluation/scores.py "$all_files" em xquad 3 results-best

eval_files="$(find runs/ -path "*seed-3/eval_results_tydiqa-goldp"  -a -not -path "*mrr*seed-3/eval_results_mlqa-test" -a -not -path "*klpw-fw*seed-3/eval_results_mlqa-test"  -a -not -path "*klpw_plus*seed-3/eval_results_mlqa-test")"
all_files="$eval_files ./runs/mbert-qa-en/eval_results_tydiqa-goldp"
echo Found $(echo "$all_files" | wc -l) evaluation files
python src/evaluation/scores.py "$all_files" em tydiqa-goldp 3 results-best
