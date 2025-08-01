import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import numpy as np


def plot_alphas(input_file, label):
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.figsize": (10, 6),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        },
    )
    df = pd.read_csv(input_file)
    x = df["Step"]
    y = df["Value"]

    # Smooth the data using Savitzky-Golay filter
    y_smooth = savgol_filter(y, window_length=51, polyorder=3)

    sns.lineplot(x=x, y=y, label=label)
    sns.lineplot(x=x, y=y_smooth, label=f"{label} (smoothed)", linestyle="--")
    for vline_x in [0, x.tolist()[-1] / 3, 2 * x.tolist()[-1] / 3, x.tolist()[-1]]:
        plt.axvline(x=vline_x, ymin=0, ymax=1, color="green", linestyle="dotted")


def get_alphas_plots(input_files):
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.figsize": (10, 6),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        },
    )
    plt.figure(figsize=(10, 6))  # Create a single figure for both plots

    colors = ["blue", "red"]

    for i, input_file in enumerate(input_files):
        name_file = os.path.basename(input_file).replace(".csv", "")
        df = pd.read_csv(input_file)
        x = df["Step"]
        y = df["Value"]

        # Calculate moving average instead of using Savitzky-Golay filter
        window_size = 51
        y_smooth = pd.Series(y).rolling(window=window_size, center=True).mean()
        # Fill NaNs at the edges
        y_smooth = y_smooth.fillna(method="ffill").fillna(method="bfill")

        # Ensure smoothed values don't exceed 1.0
        y_smooth = np.clip(y_smooth, 0, 1.0)

        # Calculate local standard deviation
        rolling_std = pd.Series(y).rolling(window=window_size, center=True).std()
        # Fill NaNs at the edges
        rolling_std = rolling_std.fillna(method="ffill").fillna(method="bfill")

        if "no-ce" in name_file:
            label = "mAP@10 Coefficient (without CE)"
        else:
            label = "mAP@10 Coefficient (with CE)"

        color = colors[i % len(colors)]

        # Plot only the smoothed line (now average)
        sns.lineplot(x=x, y=y_smooth, label=label, color=color)

        # Add standard deviation band, but clip it to stay within [0, 1]
        lower_bound = np.clip(y_smooth - rolling_std, 0, 1.0)
        upper_bound = np.clip(y_smooth + rolling_std, 0, 1.0)

        plt.fill_between(x, lower_bound, upper_bound, alpha=0.3, color=color)

    # Add vertical lines at specific points
    for vline_x in [0, x.max() / 3, 2 * x.max() / 3, x.max()]:
        plt.axvline(x=vline_x, ymin=0, ymax=1, color="green", linestyle="dotted")

    # Set y-axis limits explicitly
    plt.ylim(0, 1.0)

    # Add y-axis grid lines at regular intervals
    plt.yticks(np.arange(0, 1.1, 0.1))

    sns.set_context("notebook")
    plt.xlabel("Training steps")
    plt.ylabel("mAP@10 Coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./runs/figures/curve-MAP@k-alphas-combined.png")
    plt.show()
    plt.close()


def plot_scores(scores, label_to_line, label_to_color, label):
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.figsize": (10, 6),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        },
    )
    x = list(range(7))
    y = [scores.get(ntl, None) for ntl in x]

    sns.lineplot(
        x=x,
        y=y,
        label=label,
        linestyle=label_to_line[label],
        color=label_to_color[label],
    )


def get_models_plots(
    models, score_type, models_labels, label_to_line, label_to_color, metric
):
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.figsize": (10, 6),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        },
    )
    results_eval_dev = f"./runs/scores/results-all-{metric.lower()}-mlqa-dev-3.csv"
    df_all_dev = pd.read_csv(results_eval_dev)

    for model, model_label in zip(models, models_labels):
        model = re.sub(r"./runs.*ep-3/", "", os.path.dirname(model))
        scores = {
            ntl: float(
                df_all_dev[df_all_dev["model"] == model.replace("ntl-2", f"ntl-{ntl}")][
                    score_type
                ].iloc[0]
            )
            for ntl in range(7)
            if not ("pqt-q-random" in model and ntl == 0)
        }

        plot_scores(scores, label_to_line, label_to_color, label=model_label)

    sns.set_context("notebook")
    plt.xlabel("Number of Target Languages (ntl)")
    plt.ylabel(f"{metric} ({score_type})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"./runs/figures/curve-{metric.lower()}-{score_type}-ntl_mBERT-qa-en_mlqa-dev.png"
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    models = [
        "./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-2/ce-kl-fw-self-distil/temp-2/seed-3/",
        "./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-2/ce-kl-fw-map-coeff-self-distil/temp-2/seed-3/",
        "./runs/joint_train-xquad/en-es-de-ar-vi-hi-zh/mbert-qa-en/ep-3/ntl-2/ce/seed-3/",
    ]

    models_labels = [
        "mBERT-qa-en, skd",
        "mBERT-qa-en, skd, mAP@10",
        "mBERT-qa-en, ce",
    ]

    label_to_line = {
        "mBERT-qa-en, skd": "solid",
        "mBERT-qa-en, skd, mAP@10": "--",
        "mBERT-qa-en, ce": "solid",
    }
    label_to_color = {
        "mBERT-qa-en, skd": "b",
        "mBERT-qa-en, skd, mAP@10": "b",
        "mBERT-qa-en, ce": "r",
    }

    for metric in ["F1", "EM"]:
        for score_type in ["GXLT", "XLT"]:
            get_models_plots(
                models, score_type, models_labels, label_to_line, label_to_color, metric
            )

        input_files = [
            "./runs/scores/alpha_kl-skd.csv",
            "./runs/scores/alpha_kl-skd-no-ce.csv",
        ]

        get_alphas_plots(input_files)
