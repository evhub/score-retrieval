from __future__ import division

from collections import defaultdict

import numpy as np
from numpy.linalg import norm

from score_retrieval.constants import VECTOR_LEN
from score_retrieval.data import (
    query_paths,
    query_labels,
    database_labels,
    database_paths,
)
from score_retrieval.vec_db import (
    load_query_veclists,
    load_db_vecs,
)


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


def run_retrieval(
    query_labels=query_labels,
    query_paths=query_paths,
    database_labels=database_labels,
    database_paths=database_paths,
):
    """Run image retrieval on the given database, query."""
    q_labels, q_veclists = load_query_veclists(query_labels, query_paths)

    db_labels, db_vecs = load_db_vecs(database_labels, database_paths)

    correct = 0
    total = 0
    for correct_label, veclist in zip(q_labels, q_veclists):
        guessed_label = retrieve_veclist(veclist, db_labels, db_vecs)
        if guessed_label == correct_label:
            correct += 1
        total += 1

    acc = correct/total
    print("Got accuracy of {} for {} data points.".format(acc, total))
    return acc


if __name__ == "__main__":
    run_retrieval()
