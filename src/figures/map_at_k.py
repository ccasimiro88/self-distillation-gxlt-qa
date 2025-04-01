"""
Compute and plot the probability distributions of the teacher model "mbert-qa-en" on the training dataset XQuAD.
"""

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import json
import torch.nn.functional as F
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_probs(model, tokenizer, question, text, answer_correct):
    inputs = tokenizer.encode_plus(
        [question, text], add_special_tokens=True, return_tensors="pt"
    )
    input_ids = list(inputs["input_ids"][0].detach().numpy())
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_correct_idx = [
        input_ids.index(a)
        for a in tokenizer.encode(answer_correct, add_special_tokens=False)
    ]
    try:
        answer_start_logits, answer_end_logits, _ = model(**inputs, return_dict=False)
    except ValueError:
        answer_start_logits, answer_end_logits = model(**inputs, return_dict=False)
    # Convert to probabilities with softmax parametrazed with temperature
    temp = 4
    answer_start_probs = (
        F.softmax(answer_start_logits / temp, dim=-1)[0].detach().numpy().tolist()
    )
    answer_end_probs = (
        F.softmax(answer_end_logits / temp, dim=-1)[0].detach().numpy().tolist()
    )
    return answer_start_probs, answer_end_probs, answer_correct_idx


def plot(prob, answer_correct_idx, prob_type):
    # Set the style and parameters for high-quality figures
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
            "ytick.labelsize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        },
    )

    # Create the figure and axis objects
    plt.figure()

    # Convert probability data to a DataFrame for seaborn
    df = pd.DataFrame({"Position": range(len(prob)), "Probability": prob})

    # Plot the probability curve using seaborn
    sns.lineplot(
        data=df,
        x="Position",
        y="Probability",
        label=f"Probability of START Position",
        color="blue",
        linewidth=1.5,
    )

    # Define the delta for the shaded region
    delta = 10

    # Add vertical lines to indicate the start position
    plt.axvline(
        x=answer_correct_idx[0] - delta,
        color="green",
        linestyle="--",
        # label="Position Range",
        linewidth=1.5,
    )
    plt.axvline(
        x=answer_correct_idx[0] + delta, color="green", linestyle="--", linewidth=1.5
    )

    # Highlight the region of interest
    plt.axvspan(
        answer_correct_idx[0] - delta,
        answer_correct_idx[0] + delta,
        facecolor="green",
        alpha=0.3,
    )

    # Annotate the distance delta with a double-headed arrow
    plt.annotate(
        f"Δ",
        xy=(answer_correct_idx[0] - 20, 0.015),
        xytext=(answer_correct_idx[0] + 30, 0.0157),
        arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color="black"),
        fontsize=12,
        horizontalalignment="right",
        verticalalignment="top",
    )

    # Annotate the prob_type
    plt.annotate(
        f"{prob_type.upper()}",
        xy=(answer_correct_idx[0], 0),
        xytext=(answer_correct_idx[0] + 50, 0.007),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="black"),
        fontsize=12,
        horizontalalignment="right",
        verticalalignment="top",
    )

    # Label the axes
    plt.ylabel(f"Probability", fontsize=14)
    plt.xlabel(f"Token Positions", fontsize=14)

    # Adjust x-ticks for better readability
    xticks = [i for i in range(0, len(prob), 40)]
    xticks.append(answer_correct_idx[0])
    xticks = sorted(set(xticks), reverse=True)
    plt.xticks(xticks, fontsize=12)

    # Set y-axis limits
    plt.ylim(bottom=0)

    # Add a legend
    plt.legend(fontsize=12)

    # Improve layout and save the figure
    plt.tight_layout()
    plt.savefig(f"./runs/figures/map_at_k_plot_{prob_type}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    random.seed(1)
    model = "./runs/mbert-qa-en"

    print(f"Model: {model}")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForQuestionAnswering.from_pretrained(model)
    model.eval()

    example_vi_hi = {
        "context": "Fury d\u00f9ng c\u00e1i ch\u1ebft c\u1ee7a Coulson \u0111\u1ec3 kh\u00edch l\u1ec7 tinh th\u1ea7n nh\u00f3m si\u00eau anh h\u00f9ng v\u00e0 mong h\u1ecd s\u1ebd tr\u1edf l\u1ea1i l\u00e0m vi\u1ec7c nh\u01b0 m\u1ed9t \u0111\u1ed9i th\u1ed1ng nh\u1ea5t. Stark v\u00e0 Rogers nh\u1eadn ra r\u1eb1ng Loki mu\u1ed1n \u0111\u00e1nh b\u1ea1i c\u00e1c si\u00eau anh h\u00f9ng c\u00f4ng khai \u0111\u1ec3 kh\u1eb3ng \u0111\u1ecbnh v\u1ecb th\u1ebf th\u1ed1ng tr\u1ecb c\u1ee7a h\u1eafn tr\u00ean Tr\u00e1i \u0110\u1ea5t. H\u1eafn d\u00f9ng kh\u1ed1i Tesseract k\u1ebft h\u1ee3p thi\u1ebft b\u1ecb ti\u1ebfn s\u0129 Selvig ch\u1ebf t\u1ea1o \u0111\u1ec3 m\u1edf c\u1ed5ng t\u1eeb ph\u00eda tr\u00ean t\u00f2a th\u00e1p Stark, cho qu\u00e2n Chitauri tr\u00e0n v\u00e0o. Nh\u00f3m si\u00eau anh h\u00f9ng t\u1eadp h\u1ee3p \u0111\u1ec3 b\u1ea3o v\u1ec7 th\u00e0nh ph\u1ed1 New York (v\u1ecb tr\u00ed c\u1ee7a t\u00f2a th\u00e1p), nh\u01b0ng h\u1ecd nhanh ch\u00f3ng nh\u1eadn ra m\u00ecnh s\u1ebd b\u1ecb \u00e1p \u0111\u1ea3o trong c\u01a1n s\u00f3ng v\u0169 b\u00e3o c\u1ee7a qu\u00e2n \u0111\u1ed9i Chirauti. V\u1edbi s\u1ef1 ch\u1ec9 huy c\u1ee7a Rogers, nh\u00f3m \u0111\u00e3 ch\u1ed1ng c\u1ef1 l\u1ea1i \u0111\u01b0\u1ee3c ph\u1ea7n n\u00e0o. Banner \u0111\u00e3 tr\u1edf l\u1ea1i h\u00ecnh d\u00e1ng Hulk v\u00e0 cu\u1ed1i c\u00f9ng khu\u1ea5t ph\u1ee5c \u0111\u01b0\u1ee3c Loki. Romanoff \u0111i \u0111\u1ebfn c\u00e1nh c\u1ed5ng, n\u01a1i Ti\u1ebfn s\u0129 Selvig \u0111\u00e3 tho\u00e1t kh\u1ecfi \u0111\u01b0\u1ee3c s\u1ef1 kh\u1ed1ng ch\u1ebf c\u1ee7a Loki, \u00f4ng n\u00f3i r\u1eb1ng c\u00e2y quy\u1ec1n tr\u01b0\u1ee3ng c\u00f3 th\u1ec3 \u0111\u00f3ng c\u00e1nh c\u1ed5ng l\u1ea1i. Trong l\u00fac \u0111\u00f3, c\u1ea5p tr\u00ean c\u1ee7a Fury - H\u1ed9i \u0111\u1ed3ng An ninh Th\u1ebf gi\u1edbi, \u0111\u1ecbnh ng\u0103n ch\u1eb7n s\u1ef1 x\u00e2m l\u01b0\u1ee3c b\u1eb1ng vi\u1ec7c ph\u00f3ng \u0111\u1ea1n h\u1ea1t nh\u00e2n v\u00e0o Manhattan. Stark \u0111\u00e3 l\u00e1i t\u00ean l\u1eeda h\u1ea1t nh\u00e2n v\u00e0o c\u00e1nh c\u1ed5ng, t\u1edbi ch\u1ed7 t\u00e0u m\u1eb9 c\u1ee7a b\u1ecdn Chitauri. Chi\u1ebfc t\u00e0u m\u1eb9 b\u1ecb ph\u00e1 h\u1ee7y khi\u1ebfn t\u1ea5t c\u1ea3 Chitauri ch\u1ebft, ch\u1eb7n \u0111\u01b0\u1ee3c nguy c\u01a1 x\u00e2m l\u01b0\u1ee3c Tr\u00e1i \u0110\u1ea5t. B\u1ed9 \u00e1o gi\u00e1p c\u1ee7a Stark t\u1edbi l\u00fac n\u00e0y c\u1ea1n n\u0103ng l\u01b0\u1ee3ng v\u00e0 anh \u0111\u00e3 r\u01a1i t\u1ef1 do qua c\u00e1nh c\u1ed5ng, Hulk nhanh ch\u00f3ng c\u1ee9u \u0111\u01b0\u1ee3c Stark t\u1eeb tr\u00ean cao. Romanoff \u0111\u00f3ng c\u00e1nh c\u1ed5ng l\u1ea1i \u0111\u1ec3 b\u1ea3o v\u1ec7 Tr\u00e1i \u0110\u1ea5t. Sau c\u00f9ng, Thor \u0111\u01b0a Loki v\u00e0 kh\u1ed1i Tesseract v\u1ec1 Asgard. Fury n\u00f3i r\u1eb1ng nh\u00f3m si\u00eau anh h\u00f9ng s\u1ebd tr\u1edf l\u1ea1i n\u1ebfu c\u1ea3 th\u1ebf gi\u1edbi l\u1ea1i c\u1ea7n t\u1edbi h\u1ecd.",
        "qas": [
            {
                "id": "a6a0434b7e1c8b663fe83eb5b62db929dc4f9db6",
                "question": "\u0938\u0947\u0932\u094d\u0935\u093f\u0917 \u0928\u0947 \u0930\u094b\u092e\u0928\u0949\u092b \u0915\u094b \u0915\u094d\u092f\u093e \u092c\u0924\u093e\u092f\u093e?",
                "answers": [
                    {
                        "text": "c\u00e2y quy\u1ec1n tr\u01b0\u1ee3ng c\u00f3 th\u1ec3 \u0111\u00f3ng c\u00e1nh c\u1ed5ng l\u1ea1i",
                        "answer_start": 792,
                    }
                ],
            }
        ],
    }

    text = example_vi_hi["context"]
    answer_correct = example_vi_hi["qas"][0]["answers"][0]["text"]
    question = example_vi_hi["qas"][0]["question"]

    answer_start_probs, answer_end_probs, answer_correct_idx = get_probs(
        model, tokenizer, question, text, answer_correct
    )
    plot(answer_start_probs, answer_correct_idx, "start")
    # plot(answer_end_probs, answer_correct_idx, "end")
