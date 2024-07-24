import argparse
import json
import sys
from mlqa_evaluation_v1 import (
    metric_max_over_ground_truths as mlqa_metric_max_over_ground_truths,
)
from mlqa_evaluation_v1 import f1_score as mlqa_f1_score
from evaluate_v1 import metric_max_over_ground_truths, f1_score

from scipy.stats import wilcoxon


def mlqa_compute_ranked_sign_scores(
    dataset, predictions, context_lang, question_lang, dataset_type
):

    scores = {
        "testset": dataset_type,
        "context_lang": context_lang,
        "question_lang": question_lang,
        "f1": [],
    }
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["id"] not in predictions:
                    message = (
                        "Unanswered question " + qa["id"] + " will receive score 0."
                    )
                    print(message, file=sys.stderr)
                    continue

                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]

                if dataset_type.startswith("mlqa"):

                    f1 = (
                        mlqa_metric_max_over_ground_truths(
                            mlqa_f1_score, prediction, ground_truths, context_lang
                        )
                        * 100.0
                    )
                    scores["f1"].append(f1)
                elif dataset_type.startswith("xquad") or dataset_type.startswith(
                    "tydiqa"
                ):

                    f1 = (
                        metric_max_over_ground_truths(
                            f1_score,
                            prediction,
                            ground_truths,
                        )
                        * 100.0
                    )
                    scores["f1"].append(f1)

    return scores


if __name__ == "__main__":
    expected_version = "1.0"
    parser = argparse.ArgumentParser(
        description="Evaluation for MLQA " + expected_version
    )
    parser.add_argument("--dataset_file", help="Dataset file")
    parser.add_argument("--dataset_type", help="Dataset type (MLQA or XQUAD)")
    parser.add_argument("--prediction_file", help="Prediction File of the model")
    parser.add_argument("--model_name", help="Model name")
    parser.add_argument("--results_file", help="Results File")
    parser.add_argument("--context_language", help="Languages code of answer language")
    parser.add_argument("--question_language", help="Languages code of the question")

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

    scores = mlqa_compute_ranked_sign_scores(
        dataset,
        predictions,
        args.context_language,
        args.question_language,
        args.dataset_type,
    )

    with open(args.results_file, "a") as rs:
        json.dump(scores, rs)
        rs.write("\n")
