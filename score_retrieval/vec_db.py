import os
import pickle
from functools import partial

import numpy as np
from scipy.ndimage import imread
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
    for path, label in index_images(dataset):
        image = imread(path)
        veclist = np.asarray(map(resample, image_to_veclist_func(image)))
        assert veclist.shape[-1] == VECTOR_LEN, "{}.shape[-1] != {}".format(veclist.shape, VECTOR_LEN)
        veclist_path = os.path.splitext(path)[0] + ".npy"
        np.save(veclist, veclist_path)


def load_veclists(image_paths):
    """Yield vectors for the given image paths."""
    for path in image_paths:
        veclist_path = os.path.splitext(path)[0] + ".npy"
        veclist = np.load(veclist_path)
        assert veclist.shape[-1] == VECTOR_LEN, "{}.shape[-1] != {}".format(veclist.shape, VECTOR_LEN)
        yield veclist


def load_db_vecs(db_labels=database_labels, db_paths=database_paths):
    """Return flattened_db_labels, flattened_db_vecs."""
    flattened_db_labels = []
    flattened_db_vecs = []
    for label, veclist in zip(db_labels, load_veclists(db_paths)):
        for vec in veclist:
            flattened_db_labels.append(label)
            flattened_db_vecs.append(vec)
    return flattened_db_labels, flattened_db_vecs
