import os
import pickle
from functools import partial

import cv2
import numpy as np
from scipy import signal as ss

from benchmarks import call_benchmark
from tsai_bars import extractMeasures
from score_splitter import (
    create_waveforms,
    create_bar_waveforms,
)

from score_retrieval.constants import (
    ALG,
    arguments,
)
from score_retrieval.data import (
    index_images,
    gen_label_name_index,
    get_label,
    load_img,
    database_paths,
    query_paths,
)


def resample_arr(arr, resample_len):
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


def save_veclists(image_paths, image_to_veclist_func, grayscale=False, resample_len=None, normalize=False, debug=False):
    """Saves database of vectors using the given vector generation function."""
    for path in image_paths:
        print("Generating veclist for image {}...".format(path))

        image = load_img(path, grayscale=grayscale)
        if image is None:
            print("Got None for imread({}).".format(path))
            continue
        if debug:
            print("image.shape =", image.shape)

        raw_veclist = image_to_veclist_func(image)
        if raw_veclist is None:
            print("Got None raw_veclist from image {}.".format(path))
            continue

        veclist = []
        for vec in raw_veclist:
            if not isnull(vec):
                if resample_len is not None:
                    vec = resample_arr(vec, resample_len)
                if normalize:
                    vec = normalize_arr(vec)
                veclist.append(vec)
        veclist = np.asarray(veclist)

        if isnull(veclist):
            print("Got null veclist for {} with shape {} (raw len {}).".format(path, veclist.shape, len(raw_veclist)))
            continue
        if debug:
            print("veclist.shape =", veclist.shape)

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


def load_query_veclists(query_paths):
    """Return q_labels, q_veclists."""
    q_labels = []
    q_veclists = []
    for path in query_paths:
        veclist = load_veclist(path)
        if veclist is not None:
            label = get_label(path)
            q_labels.append(label)
            q_veclists.append(veclist)
    return q_labels, q_veclists


def load_db_vecs(database_paths):
    """Return db_labels, db_vecs, db_inds."""
    db_labels = []
    db_vecs = []
    db_indices = []

    # generate db index
    db_index = []
    for path in database_paths:
        db_index.append((get_label(path), path))

    # sort images into groups based on their order in their piece
    base_index = gen_label_name_index(db_index)
    for label, name_index in base_index.items():
        for name, paths in name_index.items():
            # for each sequence of sequential images put the vectors
            #  in the database with the right index
            i = 0
            for img_path in paths:
                veclist = load_veclist(img_path)
                if veclist is None:
                    i += 1
                else:
                    for vec in veclist:
                        db_labels.append(label)
                        db_vecs.append(vec)
                        db_indices.append(i)
                        i += 1

    return db_labels, db_vecs, db_indices


def make_benchmark_vec(image):
    resized_image = cv2.resize(image, (1024, 1024))
    return call_benchmark(images=[resized_image])


algs = {
    "bar splitting": (
        create_bar_waveforms,
        dict(grayscale=True),
    ),
    "stave splitting": (
        create_waveforms,
        dict(grayscale=True),
    ),
    "benchmark": (
        make_benchmark_vec,
        dict(),
    ),
    "new bar splitting": (
        extractMeasures,
        dict(grayscale=True),
    ),
}


if __name__ == "__main__":
    func, kwargs = algs[ALG]
    parsed_args = arguments.parse_args()
    if parsed_args.multidataset:
        paths = database_paths + query_paths
    else:
        paths = (path for path, label in index_images(parsed_args.dataset))
    save_veclists(paths, func, **kwargs)
