import os
import pickle
from functools import partial

import cv2
import numpy as np
from scipy import signal as ss

from score_retrieval.data import (
    index_images,
    database_paths,
    query_paths,
    gen_label_name_index,
    get_label,
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


def save_veclists(image_to_veclist_func, grayscale=False, resample=False, normalize=False, dataset=None):
    """Saves database of vectors using the given vector generation function."""
    for label, path in index_images(dataset):
        print("Generating veclist for image {}...".format(path))

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
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


def load_veclist(image_path):
    """Return veclist or None for the given image path."""
    veclist_path = os.path.splitext(image_path)[0] + ".npy"
    if os.path.exists(veclist_path):
        print("Loading {}...".format(veclist_path))
        return np.load(veclist_path)
    else:
        print("Skipping {}...".format(veclist_path))
        return None


def load_query_veclists(query_paths=query_paths):
    """Return q_labels, q_veclists."""
    q_labels = []
    q_veclists = []
    for path in query_paths:
        veclist = load_veclist(path)
        if veclist is not None:
            label = get_label(path)
            for i, vec in enumerate(veclist):
                q_labels.append(label)
                q_veclists.append(veclist)
    return q_labels, q_veclists


def load_db_vecs(db_paths=database_paths):
    """Return db_labels, db_vecs, db_inds."""
    db_labels = []
    db_vecs = []
    db_indices = []

    # generate db index
    db_index = []
    for path in db_paths:
        db_index.append((get_label(path), path))

    # sort images into groups based on their order in their piece
    base_index = gen_label_name_index(db_index)
    for label, name_index in base_index.items():
        for name, paths in name_index.items():

            # generate sequences of sequential images from the same label
            sequences = [[]]
            for img_path in paths:
                veclist = load_veclist(img_path)
                if veclist is None:
                    if sequences[-1]:
                        sequences.append([])
                else:
                    sequences[-1].append(veclist)

            # for each sequence put the vectors in the database with the right index
            for seq in sequences:
                i = 0
                for veclist in seq:
                    for vec in veclist:
                        db_labels.append(label)
                        db_vecs.append(vec)
                        db_indices.append(i)
                        i += 1

    return db_labels, db_vecs, db_indices


if __name__ == "__main__":
    # Our method:
    from score_splitter import create_waveforms
    save_veclists(create_waveforms, grayscale=True)

    # Benchmark method:
    # from benchmarks import call_benchmark
    # def mk_benchmark_vec(image):
    #     resized_image = cv2.resize(image, (1024, 1024))
    #     return call_benchmark(images=[resized_image])
    # save_veclists(mk_benchmark_vec)
