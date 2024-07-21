"""
Script to extact nbest predictions for evaluation
"""
import json
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.realpath(__file__))

nbest_preds = f'{script_dir}/../runs/mbert-qa-en/preds'
for file_nbest_preds in Path(nbest_preds).glob('nbest_predictions_dev*'):
    print(file_nbest_preds)
    for nbest in range(2,21):
        qids_to_answer = {}
        with open(file_nbest_preds) as fn:
            for qid, answers in json.load(fn).items():
                for k, answer in enumerate(answers):
                    if k + 1 == nbest:
                        qids_to_answer[qid] = answer['text']
                        break
        dir_preds = f'{script_dir}/../runs/mbert-qa-en/preds_nbest_{nbest}'
        os.makedirs(dir_preds, exist_ok=True)
        name_preds = os.path.basename(file_nbest_preds).replace('nbest_predictions', f'predictions_nbest_{nbest}')
        file_preds = os.path.join(dir_preds, name_preds)
        with open(file_preds, 'w') as fn:
            json.dump(qids_to_answer, fn)

