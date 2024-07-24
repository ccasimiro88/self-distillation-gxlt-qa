from scipy.stats import wilcoxon
import pandas as pd
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_a", type=str, help="File with predictions for model A")
    parser.add_argument("--file_b", type=str, help="File with predictions for model B")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--languages", help="Languages to select", nargs="+")
    parser.add_argument(
        "--do_gxlt",
        action="store_true",
        help="Allow G-XLT evaluation",
        default=False,
    )
    args = parser.parse_args()

    file_a = args.file_a
    file_b = args.file_b
    alpha = args.alpha
    languages = args.languages

    with open(file_a) as fa:
        data_a = []
        for line in fa.readlines():
            data_a.append(json.loads(line))
    with open(file_b) as fb:
        data_b = []
        for line in fb.readlines():
            data_b.append(json.loads(line))

    df_a = pd.DataFrame.from_dict(data_a)
    df_b = pd.DataFrame.from_dict(data_b)

    if languages:
        df_a = df_a[
            (df_a["context_lang"].isin(languages))
            & (df_a["question_lang"].isin(languages))
        ]
        df_b = df_b[
            (df_b["context_lang"].isin(languages))
            & (df_b["question_lang"].isin(languages))
        ]

        scores_a = df_a["f1"].apply(lambda x: list(map(float, x))).sum()
        scores_b = df_b["f1"].apply(lambda x: list(map(float, x))).sum()

    else:
        scores_a = df_a["f1"].apply(lambda x: list(map(float, x))).sum()
        scores_b = df_b["f1"].apply(lambda x: list(map(float, x))).sum()

    if not args.do_gxlt:

        df_a = df_a[df_a["context_lang"] == df_a["question_lang"]]
        df_b = df_b[df_b["context_lang"] == df_b["question_lang"]]

    # print(f"Running Wilcoxon test for:\n{file_b}\nvs\n{file_a}")

    wilcoxon_results = wilcoxon(scores_a, scores_b)
    if float(wilcoxon_results[1]) <= float(alpha):
        print("Test result is significant with p-value: {}".format(wilcoxon_results[1]))
    else:
        print(
            "Test result is not significant with p-value: {}".format(
                wilcoxon_results[1]
            )
        )
