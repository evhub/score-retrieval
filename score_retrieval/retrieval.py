from __future__ import division

from collections import defaultdict

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from fastdtw import fastdtw

from score_retrieval.eval import individual_mrr
from score_retrieval.data import (
    query_paths,
    database_paths,
    gen_data_from_args,
    gen_multi_dataset_data,
    get_label_set,
)
from score_retrieval.vec_db import (
    load_query_veclists,
    load_db_vecs,
    load_veclist,
)
from score_retrieval.constants import (
    arguments,
    LIN_WEIGHT,
    LIN_TYPE_WEIGHTS,
    TOP_N_ACCURACY,
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

            # don't do linearity if we're only doing diff
            if LIN_TYPE_WEIGHTS["diff"] < 1.0:

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

            lin_loss = 0
            for lin_type, weight in LIN_TYPE_WEIGHTS.items():
                if not weight:
                    continue
                elif lin_type == "slope":
                    lin_loss += weight * np.abs(m - 1)
                elif lin_type == "r**2":
                    lin_loss -= weight * r**2
                elif lin_type == "r":
                    lin_loss -= weight * (1 + r)/2
                elif lin_type == "diff":
                    num_greater = 0
                    prev_ind = inds[0]
                    for ind in inds[1:]:
                        if ind > prev_ind:
                            num_greater += 1
                        prev_ind = ind
                    lin_loss -= num_greater/(len(inds) - 1 if len(inds) > 1 else 1)
                else:
                    raise ValueError("unknown linearity loss type {}".format(lin_type))
            linearity_losses[label] += lin_loss

    # calculate total losses
    dist_losses = sum_min_losses/num_qvecs
    total_losses = (1 - LIN_WEIGHT) * dist_losses + LIN_WEIGHT * linearity_losses

    # order the labels
    sorted_labels = [int(label) for label in np.argsort(total_losses)]
    best_label = sorted_labels[0]
    best_label_str = label_set[best_label]

    # print debug info
    print("\tGuessed label: {}\n\t(loss: {:.5f}; dist loss: {:.5f}; lin loss: {:.5f})".format(
        best_label_str, float(total_losses[best_label]), float(dist_losses[best_label]), float(linearity_losses[best_label])))

    return sorted_labels


def run_retrieval(alg_name, query_paths=query_paths, database_paths=database_paths, debug=False):
    """Run image retrieval on the given database, query."""
    q_label_strs, q_veclists = load_query_veclists(query_paths, alg_name)
    db_label_strs, db_vecs, db_inds = load_db_vecs(database_paths, alg_name)

    label_set = get_label_set(db_label_strs)
    db_labels = [label_set.index(label) for label in db_label_strs]

    in_top_n = [0] * TOP_N_ACCURACY
    mrrs = []
    for i, (correct_label_str, veclist) in enumerate(zip(q_label_strs, q_veclists)):
        print("({}/{}) Correct label: {}".format(i+1, len(q_veclists), correct_label_str))
        correct_label = label_set.index(correct_label_str)

        # run retrieval
        sorted_labels = retrieve_veclist(veclist, db_labels, db_vecs, db_inds, label_set, debug=debug)

        # compute top N accuracy
        for i in range(len(in_top_n)):
            n = i + 1
            if correct_label in sorted_labels[:n]:
                in_top_n[i] += 1

        # compute MRR
        pos_rank = sorted_labels.index(correct_label)
        mrr = individual_mrr(pos_rank)
        print("\tMRR: {}".format(mrr))
        mrrs.append(mrr)

    for i, correct in enumerate(in_top_n):
        n = i + 1
        acc = correct/len(q_veclists)
        print("Got top {} accuracy of {} ({}/{} correct).".format(
            n, acc, correct, len(q_veclists)))

    ave_mrr = np.mean(np.array(mrrs))
    print("Got mean MRR of {}.".format(ave_mrr))

    return acc


def run_retrieval_from_args(parsed_args=None):
    """Run retrieval using the given args."""
    if parsed_args is None:
        parsed_args = arguments.parse_args()
    _data = gen_data_from_args(parsed_args)
    return run_retrieval(parsed_args.alg, query_paths=_data["query_paths"], database_paths=_data["database_paths"])


def best_vecs_for(alg_name, query_path, q_vec_ind, database_path=database_paths):
    """Helper function for visualizing the best matching vectors.

    Takes in a path to a query image and the index of the var in that
    image to get the best matches for.

    Returns an iterator of (db_image_path, index_of_matched_bar) in order
    of the best match to the worst match for the given query vector."""
    # load db
    db_label_strs, db_vecs, db_inds, db_paths = load_db_vecs(database_paths, alg_name, return_paths=True)

    # load single query vec
    q_vec = load_veclist(query_path, alg_name)[q_vec_ind]

    # compute distance vector
    dist_vec = np.squeeze(DIST_METRIC(np.array([q_vec]), np.asarray(db_vecs)))
    assert dist_vec.shape == (len(db_vecs),)

    # find best matches
    best_matches = np.argsort(dist_vec)

    # yield information about best matches in order
    for match_ind in best_matches:
        db_img_path = db_paths[match_ind]
        db_vec_ind = db_inds[match_ind]
        yield db_img_path, db_vec_ind


if __name__ == "__main__":
    run_retrieval_from_args()
