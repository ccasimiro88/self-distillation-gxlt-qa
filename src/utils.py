import json
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def qids(dataset):
    qids = []
    for data in dataset["data"]:
        for par in data["paragraphs"]:
            for qa in par["qas"]:
                qids.append(qa["id"])
    return qids


def parse_file(file):
    with open(os.path.realpath(file)) as reader:
        dataset = json.load(reader)
    return dataset


def join_datasets(files):
    dataset_joint = {"version": None, "data": []}
    datasets = []
    for file in files:
        datasets.append(parse_file(file))

    version = datasets[0]["version"]
    assert all(
        dataset["version"] == version for dataset in datasets
    ), "Trying to merge SQUAD files with different versions"

    for dataset in datasets:
        dataset_joint["data"].extend(dataset["data"])

    return dataset_joint


def kl_divergence(type, logits_stu, logits_tea, temperature):
    """Compute Kullback-Leibler distance between two probability distributions (student and teacher in this case)."""
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    if type.startswith("fw"):
        loss = loss_fct(
            F.log_softmax(logits_stu / temperature, dim=-1),
            F.softmax(logits_tea / temperature, dim=-1),
        ) * (temperature**2)
    elif type.startswith("rv"):
        loss = loss_fct(
            F.log_softmax(logits_tea / temperature, dim=-1),
            F.softmax(logits_stu / temperature, dim=-1),
        ) * (temperature**2)
    elif type.startswith("sym"):
        loss = loss_fct(
            F.log_softmax(logits_stu / temperature, dim=-1),
            F.softmax(logits_tea / temperature, dim=-1),
        ) * (temperature**2) + loss_fct(
            F.log_softmax(logits_tea / temperature, dim=-1),
            F.softmax(logits_stu / temperature, dim=-1),
        ) * (
            temperature**2
        )
    return loss


# Implementation of MAP@k from: https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code/notebook
def apk(actual, predicted, k):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mrr(predicted):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    ranks = (np.asarray(pred).nonzero()[0] for pred in predicted)
    return np.mean([1.0 / (rank[0] + 1) if rank.size else 0.0 for rank in ranks])


def compute_kl_map_at_k_coefficients(
    start_positions,
    end_positions,
    start_logits_tea,
    end_logits_tea,
    temperature,
    context=5,
    topk=10,
):
    """Compute MAP of the top-k teacher predictions based on ground-thruth positions"""

    # get ground-thruth positions with a margin context
    start_positions_true = []
    end_positions_true = []
    for start_pos, end_pos in zip(start_positions.tolist(), end_positions.tolist()):
        start_positions_true.append(range(start_pos - context, start_pos + context))
        end_positions_true.append(range(end_pos - context, end_pos + context))

    # get top-k model predictions for positions
    start_pred_idxs_topk = (
        F.softmax(start_logits_tea / temperature, dim=-1).topk(k=topk).indices.tolist()
    )
    end_preds_idxs_topk = (
        F.softmax(end_logits_tea / temperature, dim=-1).topk(k=topk).indices.tolist()
    )

    # compute top-k true predictions based on indices of the ground-thruth positions
    start_preds_topk = []
    for preds_idxs, pos_true in zip(start_pred_idxs_topk, start_positions_true):
        start_true_topk = []
        for pred_idx in preds_idxs:
            if pred_idx in pos_true:
                start_true_topk.append(1)
            else:
                start_true_topk.append(0)
        start_preds_topk.append(start_true_topk)

    end_preds_topk = []
    for preds_idxs, pos_true in zip(end_preds_idxs_topk, end_positions_true):
        end_true_topk = []
        for pred_idx in preds_idxs:
            if pred_idx in pos_true:
                end_true_topk.append(1)
            else:
                end_true_topk.append(0)
        end_preds_topk.append(end_true_topk)

    # finally, compute the MAP@k metric
    actual = [[1] for _ in range(len(start_preds_topk))]
    map_start = mapk(actual, start_preds_topk, k=topk)
    map_end = mapk(actual, end_preds_topk, k=topk)
    map_avg = (map_start + map_end) / 2
    return map_avg


def compute_kl_mrr_at_k_coefficients(
    start_positions,
    end_positions,
    start_logits_tea,
    end_logits_tea,
    temperature,
    topk=10,
):
    """Compute MAP of the top-k teacher predictions based on ground-thruth positions"""
    # get ground-thruth positions
    start_positions_true = start_positions.tolist()
    end_positions_true = end_positions.tolist()

    # get top-k model predictions for positions
    start_pred_idxs_topk = (
        F.softmax(start_logits_tea / temperature, dim=-1).topk(k=topk).indices.tolist()
    )
    end_preds_idxs_topk = (
        F.softmax(end_logits_tea / temperature, dim=-1).topk(k=topk).indices.tolist()
    )

    # compute top-k true predictions based on indices of the ground-thruth positions
    start_preds_topk = []
    for preds_idxs, pos_true in zip(start_pred_idxs_topk, start_positions_true):
        start_true_topk = []
        for pred_idx in preds_idxs:
            if pred_idx == pos_true:
                start_true_topk.append(1)
            else:
                start_true_topk.append(0)
        start_preds_topk.append(start_true_topk)

    end_preds_topk = []
    for preds_idxs, pos_true in zip(end_preds_idxs_topk, end_positions_true):
        end_true_topk = []
        for pred_idx in preds_idxs:
            if pred_idx == pos_true:
                end_true_topk.append(1)
            else:
                end_true_topk.append(0)
        end_preds_topk.append(end_true_topk)

    # finally, compute the MRR metric
    mrr_start = mrr(start_preds_topk)
    mrr_end = mrr(end_preds_topk)
    mrr_avg = (mrr_start + mrr_end) / 2
    return mrr_avg


def compute_distillation_loss(
    args,
    start_positions,
    end_positions,
    start_logits_stu,
    start_logits_tea,
    end_logits_stu,
    end_logits_tea,
    type,
):
    loss_start = kl_divergence(
        type, start_logits_stu, start_logits_tea, args.temperature
    )
    loss_end = kl_divergence(type, end_logits_stu, end_logits_tea, args.temperature)

    loss_kl = (loss_start + loss_end) / 2.0

    if args.use_map_loss_coefficients:
        alpha_kl = compute_kl_map_at_k_coefficients(
            start_positions,
            end_positions,
            start_logits_tea,
            end_logits_tea,
            args.temperature,
        )
    elif args.use_mrr_loss_coefficients:
        alpha_kl = compute_kl_mrr_at_k_coefficients(
            start_positions,
            end_positions,
            start_logits_tea,
            end_logits_tea,
            args.temperature,
        )
    else:
        alpha_kl = 1.0
    return loss_kl, alpha_kl
