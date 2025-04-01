import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style and parameters for high-quality figures
sns.set_theme(
    style="whitegrid",
    rc={
        "figure.figsize": (10, 6),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 16,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    },
)


def plot_topk_increase(df, languages):
    x = df["top_k"]

    for lang in languages:
        sns.lineplot(x=x, y=df[lang], marker="o", label=lang)

    plt.xlabel("Top-k")
    plt.ylabel("EM score")
    # plt.title("EM score as top-k Increases")
    plt.xticks(df["top_k"])
    plt.legend(title="Language")
    plt.tight_layout()

    # Saving the plot
    output_dir = "./runs/figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "topk_predictions_increase.png"))
    plt.show()
    plt.clf()


def main():
    # Sample data from the CSV
    data = {
        "top_k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "en": [43.22, 54.86, 60.78, 64.59, 67.22, 69.32, 71.04, 72.41, 73.51, 74.53],
        "es": [35.45, 46.05, 52.06, 56.10, 59.01, 61.13, 62.88, 64.48, 65.89, 67.08],
        "de": [36.73, 46.91, 52.50, 56.42, 59.27, 61.40, 63.05, 64.41, 65.67, 66.80],
        "ar": [23.66, 31.56, 36.34, 39.58, 42.10, 44.27, 46.11, 47.77, 49.01, 50.22],
        "vi": [29.06, 37.94, 43.39, 47.16, 49.78, 52.04, 53.76, 55.24, 56.73, 58.05],
        "hi": [22.12, 29.39, 33.88, 37.25, 39.78, 41.65, 43.64, 45.11, 46.46, 47.84],
        "zh": [29.99, 39.31, 44.61, 48.31, 51.25, 53.38, 55.40, 57.11, 58.46, 59.75],
    }

    df = pd.DataFrame(data)

    # Plot the increase in correct predictions for each language
    plot_topk_increase(df, df.columns[1:])


if __name__ == "__main__":
    main()
