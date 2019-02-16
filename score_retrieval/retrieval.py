from __future__ import division

from collections import defaultdict

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from fastdtw import fastdtw

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


def DTW(vec1, vec2):
    """DTW distance between two vectors."""
    return fastdtw(vec1, vec2, dist=euclidean)[0]


def L2(vec1, vec2, ignore_len_diffs=False):
    """L2 norm between two vectors."""
    if vec1.shape != vec2.shape:
        if ignore_len_diffs:
            return float("inf")
        else:
            raise ValueError("L2 vec shapes not equal: {} != {}".format(vec1.shape, vec2.shape))
    return norm(vec1 - vec2, ord=2)


DIST_METRIC = L2


def retrieve_vec(query_vec, db_labels, db_vecs, db_inds):
    """Find the value of the min dist and the index for that dist
    for each label in the database."""
    # generate dictionary mapping label to
    #  (best_dist_for_label, ind_of_the_vec_with_that_dist)
    min_scores = defaultdict(lambda: (float("inf"), None))
    for label, db_vec, ind in zip(db_labels, db_vecs, db_inds):
        dist = DIST_METRIC(db_vec, query_vec)
        if dist <= min_scores[label][0]:
            min_scores[label] = (dist, ind)
    return min_scores


LIN_SCORE_SCALE = 1
LIN_WEIGHT = 0.25


def retrieve_veclist(query_veclist, db_labels, db_vecs, db_inds):
    """Find the label with the min sum of min dist and mean change
    in index for each vector."""
    # sum best distances into dist_scores and
    #  collect all best indices into all_inds
    dist_scores = defaultdict(float)
    all_inds = defaultdict(list)
    for query_vec in query_veclist:
        min_scores = retrieve_vec(query_vec, db_labels, db_vecs, db_inds)
        for label, (vec_score, vec_ind) in min_scores.items():
            dist_scores[label] += vec_score
            all_inds[label].append(vec_ind)

    # calculate linearity by finding scale * (abs(m - 1) - r^2) of the
    #  indices (we take the negative so that smaller scores are better)
    linearity_scores = defaultdict(float)
    for label, inds in all_inds.items():

        # assume perfect linearity for veclists of length 1
        if len(inds) == 1:
            m = 1
            r = 1

        # otherwise do linear regression to determine linearity
        else:
            x_vals = np.arange(0, len(inds))
            m, b, r, p, se = linregress(x_vals, inds)
            print("m = {}, b = {}, r = {}, p = {}, se = {}".format(m, b, r, p, se))

        linearity_scores[label] += LIN_SCORE_SCALE * (np.abs(m - 1) - r**2)

    best_label = None
    best_score = float("inf")
    for label in dist_scores:

        # combine dist and linearity scores into a total score
        dist_score = dist_scores[label]
        linearity_score = linearity_scores[label]
        total_score = (1 - LIN_WEIGHT) * dist_score + LIN_WEIGHT * linearity_score
        print("total_score = {} (dist_score = {}, linearity_score = {})".format(total_score, dist_score, linearity_score))

        # best score is the smallest total_socre
        if total_score < best_score:
            best_label = label
            best_score = total_score

    print("Guessed label: {} (score: {})".format(best_label, best_score))
    return best_label


def run_retrieval(
    query_labels=query_labels,
    query_paths=query_paths,
    database_labels=database_labels,
    database_paths=database_paths,
):
    """Run image retrieval on the given database, query."""
    q_labels, q_veclists = load_query_veclists(query_labels, query_paths)

    db_labels, db_vecs, db_inds = load_db_vecs(database_labels, database_paths)

    correct = 0
    total = 0
    for correct_label, veclist in zip(q_labels, q_veclists):
        guessed_label = retrieve_veclist(veclist, db_labels, db_vecs, db_inds)
        print("Correct label was: {}".format(correct_label))
        if guessed_label == correct_label:
            correct += 1
        total += 1

    acc = correct/total
    print("Got accuracy of {} ({}/{} correct).".format(acc, correct, total))
    return acc


if __name__ == "__main__":
    run_retrieval()
