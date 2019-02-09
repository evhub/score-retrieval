from collections import defaultdict

import numpy as np
from numpy.linalg import norm
from scipy import signal as ss

from score_retrieval.constants import VECTOR_LEN


def resample(arr, resample_len=VECTOR_LEN):
    """Resample array to constant length."""
    return ss.resample(arr, resample_len)


def L2(arr1, arr2):
    """L2 norm between two vectors."""
    return norm(arr1 - arr2, ord=2)


def retrieve_vec(query_vec, database_labels, database_vecs):
    """Find the value of the min L2 for each label in the database."""
    scores = defaultdict(lambda: float("inf"))
    for label, db_vec in zip(database_labels, database_vecs):
        dist = L2(db_vec, query_vec)
        scores[label] = min(scores[label], dist)
    return scores


def retrieve_veclist(query_veclist, database_labels, database_vecs):
    """Find the label with the min sum of min L2s for each vector."""
    total_scores = defaultdict(lambda: float("inf"))
    for query_vec in query_veclist:
        scores = retrieve_vec(query_vec, database_labels, database_vecs)
        for label, vec_score in scores.items():
            total_scores[label] += vec_score

    best_label = None
    best_score = float("inf")
    for label, score in total_scores.items():
        if score < best_score:
            best_label = label
            best_score = score
    return best_label
