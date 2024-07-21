import pandas as pd
import re
import os
import matplotlib.pyplot as plt


def plot_alphas(input_file, label):
    df = pd.read_csv(input_file)
    x = df["Step"]
    # x = [x[i] for i in range(0, len(x),2)]
    y = df["Value"]
    # y = [y[i] for i in range(0, len(y),2)]

    plt.plot(x, y, label=label)
    plt.vlines(0 * x.tolist()[-1], ymin=0, ymax=1, colors="green", linestyles="dotted")
    plt.vlines(
        1 / 3 * x.tolist()[-1], ymin=0, ymax=1, colors="green", linestyles="dotted"
    )
    plt.vlines(
        2 / 3 * x.tolist()[-1], ymin=0, ymax=1, colors="green", linestyles="dotted"
    )
    plt.vlines(x.tolist()[-1], ymin=0, ymax=1, colors="green", linestyles="dotted")

    # plt.axvline(len(x))


def get_alphas_plots(input_files):
    for input_file in input_files:
        name_file = os.path.basename(input_file).replace(".csv", "")
        label = "alpha_kl"
        if "ce" in name_file:
            title = "MAP@k coefficient for SKD (no CE)"
            fig_name = "alpha_kl_no_ce"
        else:
            title = "MAP@k coefficient for SKD"
            fig_name = "alpha_kl"

        plot_alphas(input_file, label=label)

        plt.title(title)
        plt.xlabel(f"training steps")
        plt.ylabel(label)
        plt.show()
        plt.savefig(f"./runs/figures/curve-MAP@k-alphas-{fig_name}.png")
        plt.clf()


def plot_scores(scores, label_to_line, label):
    x = []
    y = []
    for ntl in range(7):
        try:
            y.append(scores[ntl])
            x.append(ntl)
        except KeyError:
            continue
    plt.plot(x, y, label_to_line[label], label=label)


def get_models_plots(models, score_type, models_labels, label_to_line, metric):
    results_eval_dev = f"./runs/results-all-{metric}-mlqa-dev-3.csv"
    df_all_dev = pd.read_csv(results_eval_dev)

    scores = {}
    for model, model_label in zip(models, models_labels):
        model = re.sub("./runs.*ep-3/", "", os.path.dirname(model))
        for ntl in range(0, 7):
            if "pqt-q-random" in model and ntl == 0:
                continue
            else:
                try:
                    gxlt_score = float(
                        df_all_dev[
                            df_all_dev["model"] == model.replace("ntl-2", f"ntl-{ntl}")
                        ][score_type]
                    )
                except TypeError:
                    import pdb

                    pdb.set_trace()
                    gxlt_score = 0
                scores[int(ntl)] = gxlt_score

        plot_scores(scores, label_to_line, label=f"{model_label}")

        plt.legend()
    plt.xlabel(f"ntl")
    plt.ylabel(f"{metric} - {score_type}")
    plt.show()
    plt.savefig(
        f"./runs/figures/curve-{metric}-{score_type}-ntl_mBERT-qa-en_mlqa-dev.png"
    )
    plt.clf()


if __name__ == "__main__":
    # Scores curves. IMPORTANT, the model paths belows are used just to get the model name, the actual model is not used.
    models = [
        "./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-2/skd/temp-2/seed-3/",
        "./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-2/skd_map/temp-2/seed-3/",
        "./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-2/ce/seed-3/",
    ]

    models_labels = [
        "mBERT-qa-en, skd",
        "mBERT-qa-en, skd + MAP@k",
        "mBERT-qa-en, ce",
    ]

    label_to_line = {
        "mBERT-qa-en, skd": "b",
        "mBERT-qa-en, skd + MAP@k": "b-.",
        "mBERT-qa-en, ce": "r",
    }
    for metric in ["f1", "em"]:
        for score_type in ["GXLT", "XLT"]:
            get_models_plots(models, score_type, models_labels, label_to_line, metric)

        # Alphas curves
        input_files = ["./runs/alpha_kl-skd.csv", "./runs/alpha_kl-skd-no-ce.csv"]

        get_alphas_plots(input_files)
