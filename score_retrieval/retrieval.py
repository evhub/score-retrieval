from __future__ import division

from collections import defaultdict

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from fastdtw import fastdtw

from score_retrieval.data import (
    query_paths,
    database_paths,
)
from score_retrieval.vec_db import (
    load_query_veclists,
    load_db_vecs,
)


def DTW(arr1, arr2):
    """DTW distance matrix between the vectors contained in the two arrays."""
    dist_arr = np.zeros((arr1.shape[0], arr2.shape[0]))
    for i, vec1 in enumerate(arr1):
        for j, vec2 in enumerate(arr2):
            dist_arr[i, j] = fastdtw(arr1, arr2, dist=euclidean)[0]
    return dist_arr


def L2(arr1, arr2):
    """Euclidean distance matrix between the vectors contained in the two arrays."""
    diff_arr = np.zeros((arr1.shape[0], arr2.shape[0], arr1.shape[1]))
    for i in range(diff_arr.shape[1]):
        diff_arr[:, i, :] = arr1
    for i in range(diff_arr.shape[0]):
        diff_arr[i, :, :] -= arr2
    return norm(diff_arr, axis=-1, ord=2)


def dot(arr1, arr2):
    """Inner product between the two arrays adjusted to look like a distance."""
    return -np.dot(arr1, arr2.T)/3


DIST_METRIC = L2
LIN_WEIGHT = 0.1
SLOPE_WEIGHT = 0.25


def retrieve_veclist(query_veclist, db_labels, db_vecs, db_inds, label_set, debug=False):
    """Find the label with the min sum of min dist and mean change
    in index for each vector."""
    # constants
    num_labels = len(label_set)
    num_qvecs = len(query_veclist)
    num_dbvecs = len(db_vecs)

    # precompute distance matrix
    dist_arr = DIST_METRIC(np.asarray(query_veclist), np.asarray(db_vecs))
    assert dist_arr.shape == (num_qvecs, num_dbvecs), "{} != {}".format(dist_arr.shape, (num_qvecs, num_dbvecs))

    # generate arrays mapping query to label to
    #  best_dist_for_label and ind_of_the_vec_with_that_dist
    min_losses = np.full((num_labels, num_qvecs), float("inf"))
    min_inds = np.zeros((num_labels, num_qvecs), dtype=int)
    for qi in range(num_qvecs):
        for dbi, (label, ind) in enumerate(zip(db_labels, db_inds)):
            dist = dist_arr[qi, dbi]
            if dist <= min_losses[label, qi]:
                min_losses[label, qi] = dist
                min_inds[label, qi] = ind

    # sum best distances into sum_min_losses
    sum_min_losses = np.sum(min_losses, axis=-1)

    # calculate linearity by finding weighted abs(m - 1) - r^2 of the
    #  indices (we take the negative so that smaller losses are better)
    linearity_losses = np.zeros(num_labels)

    # only compute linearity losses if they will be weighted
    if LIN_WEIGHT > 0:
        for label, inds in enumerate(min_inds):

            # assume perfect linearity for veclists of length 1
            if len(inds) == 1:
                m = 1
                r = 1

            # otherwise do linear regression to determine linearity
            else:
                x_vals = np.arange(0, len(inds))
                m, b, r, p, se = linregress(x_vals, inds)
                if debug:
                    print("\tm = {}, b = {}, r = {}, p = {}, se = {}".format(m, b, r, p, se))

            linearity_losses[label] += SLOPE_WEIGHT * np.abs(m - 1) - (1 - SLOPE_WEIGHT) * r**2

    # calculate total losses
    dist_losses = sum_min_losses/num_qvecs
    total_losses = (1 - LIN_WEIGHT) * dist_losses + LIN_WEIGHT * linearity_losses

    # find the best label
    best_label = int(np.argmin(total_losses))
    best_label_str = label_set[best_label]

    # print debug info
    print("\tGuessed label: {}\n\t(loss: {:.5f}; dist loss: {:.5f}; lin loss: {:.5f})".format(
        best_label_str, float(total_losses[best_label]), float(dist_losses[best_label]), float(linearity_losses[best_label])))

    return best_label_str


def run_retrieval(query_paths=query_paths, database_paths=database_paths, debug=False):
    """Run image retrieval on the given database, query."""
    q_label_strs, q_veclists = load_query_veclists(query_paths)
    db_label_strs, db_vecs, db_inds = load_db_vecs(database_paths)

    label_set = list(set(db_label_strs))
    db_labels = [label_set.index(label) for label in db_label_strs]

    correct = 0
    for i, (correct_label, veclist) in enumerate(zip(q_label_strs, q_veclists)):
        print("({}/{}) Correct label: {}".format(i+1, len(q_veclists), correct_label))
        guessed_label = retrieve_veclist(veclist, db_labels, db_vecs, db_inds, label_set, debug=debug)
        if guessed_label == correct_label:
            correct += 1

    acc = correct/len(q_veclists)
    print("Got accuracy of {} ({}/{} correct).".format(acc, correct, len(q_veclists)))
    return acc


if __name__ == "__main__":
    run_retrieval()
