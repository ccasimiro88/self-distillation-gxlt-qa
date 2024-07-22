from scipy.stats import wilcoxon
import sys

if __name__ == "__main__":
    file_a = sys.argv[1]
    file_b = sys.argv[2]

    alpha = 0.05

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
