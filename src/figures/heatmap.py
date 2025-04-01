import seaborn as sns
import pandas as pd
import json
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import os
import math
import re
from argparse import ArgumentParser

# Set the style and parameters for high-quality figures
sns.set_theme()
plt.rcParams.update(
    {
        "figure.figsize": (10, 8),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }
)


def score_question_context_matrix(file, langs, score_type):
    lines = []
    with open(file) as fn:
        for line in fn.readlines():
            lines.append(dict(json.loads(line.replace("'", '"'))))
    df = pd.DataFrame.from_records(lines)
    df = df.rename(columns={"exact_match": "em"})

    languages_order = CategoricalDtype(langs, ordered=True)
    df["context_lang"] = df["context_lang"].astype(languages_order)
    df["question_lang"] = df["question_lang"].astype(languages_order)
    df = df.sort_values(["question_lang", "context_lang"])
    df = df.pivot(index="question_lang", columns="context_lang", values=score_type)

    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--testset", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--heatmap_type", type=str, default="question-context-gxlt")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(script_dir, "../../runs/figures")
    os.makedirs(output_dir, exist_ok=True)

    testset = args.testset
    file = args.file
    suffix = str(args.suffix)
    heatmap_type = args.heatmap_type
    metric = args.metric

    file_ref = f"./runs/mbert-qa-en/eval_results_{testset.lower()}"
    try:
        file_ref_all_dev = f"./runs/scores/results-all-{metric.lower()}-mlqa-dev-3.csv"
    except AttributeError:
        import pdb; pdb.set_trace()
    langs = "en es de ar vi hi zh".split()

    if heatmap_type == "question-context-gxlt":
        # compute reference heatmap (mBERT-qa-en) and print it
        df_ref = score_question_context_matrix(file_ref, langs, metric.lower())
        min_value = math.floor(df_ref.to_numpy().min())
        max_value = math.ceil(df_ref.to_numpy().max())
        sns.heatmap(
            df_ref,
            annot=True,
            cmap="Blues",
            fmt=".1f",
            vmin=min_value,
            vmax=max_value,
            cbar=False  # Add this line to hide the color bar
        )
        plt.title(f"mBERT-qa-en, ZS")
        plt.xlabel("Context Language")
        plt.ylabel("Question Language")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/heatmap-gxlt_mBERT-qa-en_{testset.lower()}_{metric.lower()}.png"
        )
        plt.show()
        plt.clf()

        df = score_question_context_matrix(file, langs, metric.lower())
        df_dif = df - df_ref
        sns.heatmap(
            df_dif,
            annot=True,
            cmap=sns.cm.rocket_r,
            fmt=".1f",
            vmin=-5,
            vmax=22,
            cbar=False  # Add this line to hide the color bar
        )
        plt.title(f"{suffix}")
        plt.xlabel("Context Language")
        plt.ylabel("Question Language")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/heatmap-gxlt_{suffix.lower().replace(' ', '').replace('+', ',')}_{testset.lower()}_{metric.lower()}.png"
        )
        plt.show()
        plt.clf()

    elif heatmap_type == "temp-vs-ntl":
        df_all_dev = pd.read_csv(file_ref_all_dev)

        scores = []
        for file in args.files:
            model = re.sub("./runs.*ep-3/", "", os.path.dirname(file))
            score = float(df_all_dev[df_all_dev["model"] == model]["GXLT"])
            temp = re.search("(temp-[0-9]+)", model).group().split("-")[1]
            ntl = re.search("(ntl-[0-9])", model).group().split("-")[1]
            score = {"f1_gxlt": score, "temp": temp, "ntl": ntl}
            scores.append(score)
        df = pd.DataFrame.from_records(scores)
        temp_order = ["18", "8", "4", "2"]
        df["temp"] = df["temp"].astype(str)
        df["temp"] = pd.Categorical(df["temp"], categories=temp_order)
        df.sort_values(["temp", "ntl"])
        df = df.pivot(index="temp", columns="ntl", values="f1_gxlt")
        vmin = round(min([s["f1_gxlt"] for s in scores]), 2)
        vmax = round(max([s["f1_gxlt"] for s in scores]), 2)
        sns.heatmap(
            df,
            annot=True,
            cmap=sns.cm.rocket_r,
            fmt=".2f",
            vmin=vmin,
            vmax=vmax,
            cbar=False  # Add this line to hide the color bar
        )
        plt.title(f"{suffix}")
        plt.xlabel("Number of Target Languages (ntl)")
        plt.ylabel("Temperature (t)")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/heatmap-temp-ntl-gxlt-{suffix.lower().replace(' ', '').replace('+', ',')}_{testset.lower()}_{metric.lower()}.png"
        )
        plt.show()
        plt.clf()
