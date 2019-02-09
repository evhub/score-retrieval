import os
import pickle
from functools import partial

import numpy as np
from scipy.ndimage import imread

from score_retrieval.data import (
    index_images,
    database_paths,
    query_paths,
)
from score_retrieval.constants import VECTOR_LEN


def save_vecs(image_to_vec_func, dataset=None):
    for path, label in index_images(dataset):
        image = imread(path)
        vec = image_to_vec_func(image)
        assert vec.shape == (VECTOR_LEN,), "{}.shape != ({},)".format(vec.shape, VECTOR_LEN)
        vec_path = os.path.splitext(path)[0] + ".pickle"
        pickle.dump((label, vec), vec_path, protocol=pickle.HIGHEST_PROTOCOL)


def load_vecs(image_paths):
    for path in image_paths:
        pickle_path = os.path.splitext(path)[0] + ".pickle"
        yield pickle.load(pickle_path, protocol=pickle.HIGHEST_PROTOCOL)


get_database_vecs = partial(load_vecs, database_paths)

get_query_vecs = partial(load_vecs, query_paths)
