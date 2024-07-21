from distutils.log import debug
import pandas as pd
import json
import sys
from collections import defaultdict
import os
import re


files = sys.argv[1]
metric = sys.argv[2]
testset = sys.argv[3]
seed = sys.argv[4]
output_filename = sys.argv[5]

print(files)
languages = {
    "mlqa-dev": "en es de ar vi hi zh".split(),
    "mlqa-test": "en es de ar vi hi zh".split(),
    "tydiqa-goldp": "bengali finnish indonesian korean russian swahili telugu".split(),  # zero-shot
    "xquad": "el ru tr th".split(),
}  # zero-shot

scores = defaultdict(list)
for file in files.split():
    records = []
    with open(file) as fn:
        for line in fn.readlines():
            records.append(dict(json.loads(line.replace("'", '"'))))

    df = pd.DataFrame.from_dict(records)
    df = df.rename(columns={"exact_match": "em"})
    df["em"] = df["em"].astype(float)
    df["f1"] = df["f1"].astype(float)

    # Aggregated scores
    model = os.path.dirname(file)

    model = re.sub("runs.*ep-3/", "", model)
    scores["model"].append(model)
    if not testset in ["tydiqa-goldp", "xquad"]:
        scores[f"GXLT"].append(round(df[metric].mean(), 2))

    scores[f"XLT"].append(
        round(
            df[
                (df["context_lang"] == df["question_lang"])
                & (df["context_lang"].isin(languages[testset]))
                & (df["question_lang"].isin(languages[testset]))
            ][metric].mean(),
            2,
        )
    )
    for language in languages[testset]:
        mean = df[(df["context_lang"] == language) & (df["question_lang"] == language)][
            metric
        ].mean()

        scores[f"{language}-mean"].append(round(mean, 2))
print(scores)

pd.DataFrame.from_dict(scores).to_csv(
    f"runs/{output_filename}-{metric}-{testset}-{seed}.csv",
    mode="w",
    index=False,
)
