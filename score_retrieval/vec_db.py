import os
import pickle
from functools import partial

import cv2
import numpy as np
from scipy import signal as ss

from score_retrieval.data import (
    index_images,
    database_labels,
    database_paths,
    query_labels,
    query_paths,
)
from score_retrieval.constants import VECTOR_LEN


def resample_arr(arr, resample_len=VECTOR_LEN):
    """Resample array to constant length."""
    out_arr = ss.resample(np.asarray(arr), resample_len)
    assert out_arr.shape == (resample_len,), "{}.shape == {} != ({},)".format(out_arr, out_arr.shape, resample_len)
    return out_arr


def normalize_arr(arr):
    """Normalize array to constant mean and stdev."""
    return (arr - np.mean(arr))/np.std(arr)


def isnull(arr):
    """Determine whether the given array is null."""
    return not arr.shape or sum(arr.shape) == 0


def save_veclists(image_to_veclist_func, resample=False, normalize=False, dataset=None):
    """Saves database of vectors using the given vector generation function."""
    for label, path in index_images(dataset):
        print("Generating veclist for image {}...".format(path))

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print("Got None for imread({}).".format(path))
            continue

        raw_veclist = image_to_veclist_func(image)
        if raw_veclist is None:
            print("Got None raw_veclist from image {}.".format(path))
            continue

        veclist = []
        for vec in raw_veclist:
            if not isnull(vec):
                if resample:
                    vec = resample_arr(vec)
                if normalize:
                    vec = normalize_arr(vec)
                veclist.append(vec)
        veclist = np.asarray(veclist)
        print("veclist.shape =", veclist.shape)

        if isnull(veclist):
            print("Got null veclist for {} with shape {} (raw len {}).".format(path, veclist.shape, len(raw_veclist)))
            continue

        veclist_path = os.path.splitext(path)[0] + ".npy"
        np.save(veclist_path, veclist)


def load_veclists(image_labels, image_paths):
    """Yield (label, veclist) for the given image paths."""
    for label, path in zip(image_labels, image_paths):
        veclist_path = os.path.splitext(path)[0] + ".npy"
        if os.path.exists(veclist_path):
            print("Loading {}...".format(veclist_path))
            veclist = np.load(veclist_path)
            yield label, veclist
        else:
            print("Skipping {}...".format(veclist_path))


def load_query_veclists(query_labels=query_labels, query_paths=query_paths):
    """Return query_labels, query_vecs."""
    q_labels = []
    q_veclists = []
    for label, veclist in load_veclists(query_labels, query_paths):
        q_labels.append(label)
        q_veclists.append(veclist)
    return q_labels, q_veclists


def load_db_vecs(db_labels=database_labels, db_paths=database_paths):
    """Return db_labels, db_vecs, db_inds."""
    flattened_db_labels = []
    flattened_db_vecs = []
    flattened_db_indices = []
    for label, veclist in load_veclists(db_labels, db_paths):
        for i, vec in enumerate(veclist):
            flattened_db_labels.append(label)
            flattened_db_vecs.append(vec)
            flattened_db_indices.append(i)
    return flattened_db_labels, flattened_db_vecs, flattened_db_indices


if __name__ == "__main__":
    from score_splitter import create_waveforms
    save_veclists(create_waveforms)
    # from benchmarks import call_benchmark
    # def mk_benchmark_vec(image):
    #     resized_image = cv2.resize(image, (1024, 1024))
    #     return call_benchmark(images=[resized_image])
    # save_veclists(mk_benchmark_vec)
