from scipy.stats import wilcoxon
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_a", type=str, help="File with predictions for model A")
    parser.add_argument("--file_b", type=str, help="File with predictions for model B")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")

    args = parser.parse_args()
    file_a = args.file_a
    file_b = args.file_b

    alpha = args.alpha

    with open(file_a) as fa:
        data_a = [float(data.strip("\n")) for data in fa.readlines()]

    with open(file_b) as fb:
        data_b = [float(data.strip("\n")) for data in fb.readlines()]

    wilcoxon_results = wilcoxon(data_a, data_b)
    if float(wilcoxon_results[1]) <= float(alpha):
        print(
            "\nTest result is significant with p-value: {}".format(wilcoxon_results[1])
        )
    else:
        print(
            "\nTest result is not significant with p-value: {}".format(
                wilcoxon_results[1]
            )
        )
