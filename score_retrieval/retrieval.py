from __future__ import division

from collections import defaultdict

import numpy as np
from numpy.linalg import norm
from scipy import signal as ss

from score_retrieval.constants import VECTOR_LEN


def resample(arr, resample_len=VECTOR_LEN):
    """Resample array to constant length."""
    return ss.resample(arr, resample_len)


def L2(vec1, vec2):
    """L2 norm between two vectors."""
    assert vec1.shape == (VECTOR_LEN,), "{}.shape != ({},)".format(vec1.shape, VECTOR_LEN)
    assert vec2.shape == (VECTOR_LEN,), "{}.shape != ({},)".format(vec2.shape, VECTOR_LEN)
    return norm(vec1 - vec2, ord=2)


def retrieve_vec(query_vec, db_labels, db_vecs):
    """Find the value of the min L2 for each label in the database."""
    scores = defaultdict(lambda: float("inf"))
    for label, db_vec in zip(db_labels, db_vecs):
        dist = L2(db_vec, query_vec)
        scores[label] = min(scores[label], dist)
    return scores


def retrieve_veclist(query_veclist, db_labels, db_vecs):
    """Find the label with the min sum of min L2s for each vector."""
    total_scores = defaultdict(lambda: float("inf"))
    for query_vec in query_veclist:
        scores = retrieve_vec(query_vec, db_labels, db_vecs)
        for label, vec_score in scores.items():
            total_scores[label] += vec_score

    best_label = None
    best_score = float("inf")
    for label, score in total_scores.items():
        if score < best_score:
            best_label = label
            best_score = score
    return best_label


if __name__ == "__main__":
    from score_retrieval.data import (
        query_labels,
    )
    from score_retrieval.vec_db import (
        get_query_veclists,
        load_db_labels_vecs,
    )

    query_veclists = get_query_veclists()

    db_labels, db_vecs = load_db_labels_vecs()

    correct = 0
    total = 0
    for correct_label, veclist in zip(query_labels, query_veclists):
        guessed_label = retrieve_veclist(veclist, db_labels, db_vecs)
        if guessed_label == correct_label:
            correct += 1
        total += 1

    acc = correct/total
    print("Got accuracy of {} for {} data points.".format(acc, total))
