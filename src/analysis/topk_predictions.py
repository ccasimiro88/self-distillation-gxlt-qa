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

# Initialize dictionaries to store correct predictions and total answers per language
preds_correct_topk = defaultdict(lambda: defaultdict(int))
total_answers_per_lang = defaultdict(int)

mbert_nbest_preds_dir = f"{script_dir}/../../runs/mbert-qa-en/preds"
# Aggregate predictions from the MLQA test split
total_qa_examples = defaultdict(int)
for nbest_pred_file in Path(mbert_nbest_preds_dir).glob("*nbest_predictions_test*"):
    dataset_name = nbest_pred_file.name.split("nbest_predictions_")[1]
    question_lang = dataset_name.replace(".json", "").split("question-")[1]
    preds = json.load(open(nbest_pred_file))

    for pred_id in preds:
        total_qa_examples[question_lang] += 1
        for topk, pred in enumerate(preds[pred_id]):
            if pred["text"] == answer_correct[dataset_name][pred_id]["text"]:
                # Increment correct prediction count for all k >= topk + 1
                for k in range(topk + 1, len(preds[pred_id]) + 1):
                    preds_correct_topk[k][question_lang] += 1
                break

# Calculate the percentage of correct predictions for each top-k
preds_correct_topk_percent = defaultdict(dict)
for topk in preds_correct_topk:
    for lang in preds_correct_topk[topk]:
        preds_correct_topk_percent[topk][lang] = round(
            (preds_correct_topk[topk][lang] / total_qa_examples[lang]) * 100, 2
        )

# Print and save the results
pprint(preds_correct_topk_percent)

df = pd.DataFrame.from_dict(preds_correct_topk_percent, orient="index").sort_index()
df.to_csv(f"{script_dir}/../../runs/topk_predictions.csv")
