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
)
from score_retrieval.constants import VECTOR_LEN


def resample(arr, resample_len=VECTOR_LEN):
    """Resample array to constant length."""
    return ss.resample(np.asarray(arr), resample_len)


def save_veclists(image_to_veclist_func, dataset=None):
    """Saves database of vectors using the given vector generation function."""
    for label, path in index_images(dataset):
        print("Generating veclist for image {}...".format(path))
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert image is not None, "imread({}) is None".format(path)
        raw_veclist = image_to_veclist_func(image)
        veclist = np.asarray(map(resample, raw_veclist))
        if veclist.shape:
            assert veclist.shape[-1] == VECTOR_LEN, "{}.shape[-1] != {}".format(veclist.shape, VECTOR_LEN)
            veclist_path = os.path.splitext(path)[0] + ".npy"
            np.save(veclist, veclist_path)
        else:
            print("Got null veclist for {} with shape {} (raw shape {}).".format(path, veclist.shape, raw_veclist.shape))


def load_veclists(image_labels, image_paths):
    """Yield (label, veclist) for the given image paths."""
    for label, path in zip(image_labels, image_paths):
        veclist_path = os.path.splitext(path)[0] + ".npy"
        if os.path.exists(veclist_path):
            print("Loading {}...".format(veclist_path))
            veclist = np.load(veclist_path)
            assert veclist.shape[-1] == VECTOR_LEN, "{}.shape[-1] != {}".format(veclist.shape, VECTOR_LEN)
            yield label, veclist
        else:
            print("Skipping {}...".format(veclist_path))


def load_db_vecs(db_labels=database_labels, db_paths=database_paths):
    """Return flattened_db_labels, flattened_db_vecs."""
    flattened_db_labels = []
    flattened_db_vecs = []
    for label, veclist in load_veclists(db_paths):
        for vec in veclist:
            flattened_db_labels.append(label)
            flattened_db_vecs.append(vec)
    return flattened_db_labels, flattened_db_vecs


if __name__ == "__main__":
    from score_splitter import create_waveforms
    save_veclists(create_waveforms)
