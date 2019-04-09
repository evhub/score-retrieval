from __future__ import division

import numpy as np

from score_retrieval.data import (
    query_labels,
    database_labels,
    database_paths,
    label_set,
)


def get_labels(indices_by_label):
    """Turn indices by label into labels."""
    labels = []
    for label, indices in enumerate(indices_by_label):
        for ind in indices:
            if len(labels) - 1 < ind:
                labels += [None] * (ind - len(labels) + 1)
            labels[ind] = label
    return labels


def get_all_pos_ranks(db_rankings, q_labels=None, db_labels=None):
    """
    db_rankings[i, j] = index in db_labels of the
        (i+1)th ranked database image for the (j+1)th query

    returns: generator of lists of rankings starting
        from 0 of positive labels for each query
    """
    if db_labels is None:
        db_labels = [label_set.index(label) for label in database_labels]
    if q_labels is None:
        q_labels = [label_set.index(label) for label in query_labels]
    query_rankings = db_rankings.T
    for query_index, query_label in enumerate(q_labels):
        # first rank all the labels
        ranked_labels = []
        for database_index in query_rankings[query_index]:
            label = db_labels[database_index]
            if label not in ranked_labels:
                ranked_labels.append(label)

        # then yield an array of just the rank of the correct label
        pos_rank = ranked_labels.index(query_label)
        yield np.array([pos_rank])


def calculate_mrr(all_pos_ranks):
    """Compute average MRR for the given pos_ranks."""
    mrrs = []
    for pos_ranks in all_pos_ranks:
        if len(pos_ranks):
            mrrs.append(np.mean(individual_mrr(pos_ranks)))
    return np.mean(np.array(mrrs))


def individual_mrr(pos_ranks):
    """Compute a single MRR from the given pos_ranks."""
    return 1/(pos_ranks + 1)


def calculate_acc(all_pos_ranks, top_n=1):
    """Compute top-n accuracy for the given pos_ranks."""
    total = 0.0
    correct = 0.0
    for pos_ranks in all_pos_ranks:
        if len(pos_ranks):
            total += 1
            for i in range(top_n):
                if i in pos_ranks:
                    correct += 1
                    break
    acc = correct / total
    return acc, correct, total


def individual_ap(pos_ranks):
    """Compute a single AP from the given pos_ranks."""
    # handle single integers
    if isinstance(pos_ranks, int):
        return np.mean(individual_ap(np.array([pos_ranks])))

    # handle arrays
    ap = 0
    recall_step = 1 / len(pos_ranks)
    for i, rank in enumerate(pos_ranks):
        precision_0 = 1 if rank == 0 else i/rank
        precision_1 = (i + 1)/(rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2
    return ap


def calculate_map(all_pos_ranks):
    """Compute MAP for the given pos_ranks."""
    aps = []
    for pos_ranks in all_pos_ranks:
        if len(pos_ranks):
            aps.append(np.mean(individual_ap(pos_ranks)))
    return np.mean(np.array(aps))
