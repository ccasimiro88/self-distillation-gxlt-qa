import json
import os
from pathlib import Path
from collections import defaultdict
from pprint import pprint
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

# Store correct answers for each file in the MLQA-dev split
mlqa_dir = f"{script_dir}/../../corpora/MLQA_V1"
answer_correct = dict()
for file in Path(mlqa_dir).rglob("*.json"):
    filename = file.name
    answer_correct[filename] = defaultdict()
    dataset = json.load(open(file))
    for data in dataset["data"]:
        for par in data["paragraphs"]:
            for qa in par["qas"]:
                answer_correct[filename][qa["id"]] = qa["answers"][0]

preds_correct_topk = defaultdict(lambda: defaultdict(int))
mbert_nbest_preds_dir = f"{script_dir}/../../runs/mbert-qa-en/preds"
# aggregate predictions from both MLQA dev and test split
nbest_pred_files = list(Path(mbert_nbest_preds_dir).glob("*nbest_predictions_dev*"))
nbest_pred_files.extend(Path(mbert_nbest_preds_dir).glob("*nbest_predictions_test*"))
for nbest_pred_file in Path(mbert_nbest_preds_dir).glob("*nbest_predictions_dev*"):
    dataset_name = nbest_pred_file.name.split("nbest_predictions_")[1]
    question_lang = dataset_name.replace(".json", "").split("question-")[1]
    preds = json.load(open(nbest_pred_file))
    for pred_id in preds:
        for topk, pred in enumerate(preds[pred_id]):
            if pred["text"] == answer_correct[dataset_name][pred_id]["text"]:
                preds_correct_topk[topk + 1][question_lang] += 1
                break

total_answers_per_lang = defaultdict(int)
for topk in preds_correct_topk:
    for lang in preds_correct_topk[topk]:
        total_answers_per_lang[lang] += preds_correct_topk[topk][lang]

for topk in preds_correct_topk:
    preds_correct_topk[topk] = dict(preds_correct_topk[topk])
    for lang in preds_correct_topk[topk]:
        # preds_correct_topk[topk][lang] = round((preds_correct_topk[topk][lang]/total_answers_per_lang[lang]) * 100, 2)
        preds_correct_topk[topk][lang] = preds_correct_topk[topk][lang]
pprint(preds_correct_topk)

df = pd.DataFrame.from_dict(preds_correct_topk, orient="index").sort_index()
df.to_csv(f"{script_dir}/../../runs/correct_answers_lang_topk.csv")
