# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
""" Slightly adapted version official evaluation script for the MLQA dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import unicodedata

PUNCT = {
    chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
}.union(string.punctuation)
WHITESPACE_LANGS = ["en", "es", "hi", "vi", "de", "ar"]
MIXED_SEGMENTATION_LANGS = ["zh"]


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r"[\u4e00-\u9fa5]", char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, lang):
        if lang == "en":
            return re.sub(r"\b(a|an|the)\b", " ", text)
        elif lang == "es":
            return re.sub(r"\b(un|una|unos|unas|el|la|los|las)\b", " ", text)
        elif lang == "hi":
            return text  # Hindi does not have formal articles
        elif lang == "vi":
            return re.sub(r"\b(của|là|cái|chiếc|những)\b", " ", text)
        elif lang == "de":
            return re.sub(
                r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
                " ",
                text,
            )
        elif lang == "ar":
            return re.sub("\sال^|ال", " ", text)
        elif lang == "zh":
            return text  # Chinese does not have formal articles
        # Do not apply any processing for unknown languages
        return text

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            # "Apply white-space segmentation for unknown languages"
            tokens = whitespace_tokenize(text)
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)


def f1_score(prediction, ground_truth, lang):
    prediction_tokens = normalize_answer(prediction, lang).split()
    ground_truth_tokens = normalize_answer(ground_truth, lang).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, lang):
    return normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, lang):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, lang)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, context_lang, question_lang, dataset_type):
    f1 = exact_match = total = 0

    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = (
                        "Unanswered question " + qa["id"] + " will receive score 0."
                    )
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths, context_lang
                )
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths, context_lang
                )

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    eval_report = {
        "testset": dataset_type,
        "context_lang": context_lang,
        "question_lang": question_lang,
        "f1": f1,
        "exact_match": exact_match,
    }
    return eval_report


def compute_ranked_sign_scores(
    dataset, predictions, context_lang, question_lang, dataset_type
):

    signed_rank_scores = {"f1": [], "exact_match": []}
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["id"] not in predictions:
                    message = (
                        "Unanswered question " + qa["id"] + " will receive score 0."
                    )
                    print(message, file=sys.stderr)
                    continue
                import pdb

                pdb.set_trace()
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match = (
                    metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths, context_lang
                    )
                    * 100.0
                )
                signed_rank_scores["exact_match"].append(exact_match)
                f1 = (
                    metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths, context_lang
                    )
                    * 100.0
                )
                signed_rank_scores["f1"].append(f1)

    return signed_rank_scores


if __name__ == "__main__":
    expected_version = "1.0"
    parser = argparse.ArgumentParser(
        description="Evaluation for MLQA " + expected_version
    )
    parser.add_argument("--dataset_file", help="Dataset file")
    parser.add_argument("--dataset_type", help="Dataset type (MLQA or XQUAD)")
    parser.add_argument("--prediction_file", help="Prediction File")
    parser.add_argument("--results_file", help="Results File")
    parser.add_argument("--context_language", help="Language code of answer language")
    parser.add_argument("--question_language", help="Language code of the question")

    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if str(dataset_json["version"]) != expected_version:
            print(
                "Evaluation expects v-"
                + expected_version
                + ", but got dataset with v-"
                + dataset_json["version"],
                file=sys.stderr,
            )
        dataset = dataset_json["data"]

    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    with open(args.results_file, "a") as fn:
        fn.write(
            str(
                evaluate(
                    dataset,
                    predictions,
                    args.context_language,
                    args.question_language,
                    args.dataset_type,
                )
            )
            + "\n"
        )
