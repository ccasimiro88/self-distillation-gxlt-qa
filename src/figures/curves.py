import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Set the style and parameters for high-quality figures
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
)


def plot_alphas(input_file, label):
    df = pd.read_csv(input_file)
    x = df["Step"]
    y = df["Value"]

    # Smooth the data using Savitzky-Golay filter
    y_smooth = savgol_filter(y, window_length=51, polyorder=3)

    plt.plot(x, y, label=label)
    plt.plot(x, y_smooth, label=f"{label} (smoothed)", linestyle="--")
    plt.vlines(
        [0, x.tolist()[-1] / 3, 2 * x.tolist()[-1] / 3, x.tolist()[-1]],
        ymin=0,
        ymax=1,
        colors="green",
        linestyles="dotted",
    )


def get_alphas_plots(input_files):
    for input_file in input_files:
        name_file = os.path.basename(input_file).replace(".csv", "")
        label = "alpha_kl"
        if "no-ce" in name_file:
            title = "MAP@k coefficient for SKD (no CE)"
            fig_name = "alpha_kl_no_ce"
        else:
            title = "MAP@k coefficient for SKD"
            fig_name = "alpha_kl"

        plot_alphas(input_file, label=label)

        plt.title(title)
        plt.xlabel("Training steps")
        plt.ylabel(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./runs/figures/curve-MAP@k-alphas-{fig_name}.png")
        plt.show()
        plt.clf()


def plot_scores(scores, label_to_line, label):
    x = list(range(7))
    y = [scores.get(ntl, None) for ntl in x]

    plt.plot(x, y, label_to_line[label], label=label)


def get_models_plots(models, score_type, models_labels, label_to_line, metric):
    results_eval_dev = f"./runs/scores/results-all-{metric.lower()}-mlqa-dev-3.csv"
    df_all_dev = pd.read_csv(results_eval_dev)

    for model, model_label in zip(models, models_labels):
        model = re.sub(r"./runs.*ep-3/", "", os.path.dirname(model))
        scores = {
            ntl: float(
                df_all_dev[df_all_dev["model"] == model.replace("ntl-2", f"ntl-{ntl}")][
                    score_type
                ]
            )
            for ntl in range(7)
            if not ("pqt-q-random" in model and ntl == 0)
        }

        plot_scores(scores, label_to_line, label=model_label)

    plt.xlabel("ntl")
    plt.ylabel(f"{metric} ({score_type})")
    # plt.title(f"{metric} {score_type} vs ntl")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"./runs/figures/curve-{metric.lower()}-{score_type}-ntl_mBERT-qa-en_mlqa-dev.png"
    )
    plt.show()
    plt.clf()


if __name__ == "__main__":
    models = [
        "./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-2/ce-kl-fw-self-distil/temp-2/seed-3/",
        "./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-2/ce-kl-fw-map-coeff-self-distil/temp-2/seed-3/",
        "./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-2/ce/seed-3/",
    ]

    models_labels = [
        "mBERT-qa-en, skd",
        "mBERT-qa-en, skd, mAP@k",
        "mBERT-qa-en, ce",
    ]

    label_to_line = {
        "mBERT-qa-en, skd": "b",
        "mBERT-qa-en, skd, mAP@k": "b-.",
        "mBERT-qa-en, ce": "r",
    }

    for metric in ["F1", "EM"]:
        for score_type in ["GXLT", "XLT"]:
            get_models_plots(models, score_type, models_labels, label_to_line, metric)

        input_files = [
            "./runs/scores/alpha_kl-skd.csv",
            "./runs/scores/alpha_kl-skd-no-ce.csv",
        ]

        get_alphas_plots(input_files)
