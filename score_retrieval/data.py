import os
import random
from collections import defaultdict

from scipy.ndimage import imread

from score_retrieval.constants import (
    DATA_DIR,
    IMG_EXT,
    SKIP_QUERYLESS,
    SAMPLE,
)


def index_images(dataset=None):
    """Return an iterator of (label, path) for all images."""
    data_dir = DATA_DIR
    if dataset is not None:
        data_dir = os.path.join(data_dir, dataset)
    for dirpath, _, filenames in os.walk(data_dir):
        for fname in filenames:
            if os.path.splitext(fname)[-1] == IMG_EXT:
                yield dirpath, os.path.join(dirpath, fname)


def index_by_label(dataset=None):
    """Return dictionary mapping labels to image paths."""
    index = defaultdict(list)
    for label, img_path in index_images(dataset):
        index[label].append(img_path)
    return index


def index_data(index=None, dataset=None):
    """Return database_paths, database_labels, query_paths, query_labels lists."""
    if index is None:
        index = index_by_label(dataset)
    database_paths = []
    database_labels = []
    query_paths = []
    query_labels = []
    for label, img_paths in index.items():
        if len(img_paths) < 2:
            if SKIP_QUERYLESS:
                continue
            head, tail = img_paths, []
        else:
            head, tail = img_paths[:-1], img_paths[-1]
        database_paths += head
        database_labels += [label]*len(head)
        query_paths.append(tail)
        query_labels.append(label)
    return database_paths, database_labels, query_paths, query_labels


def sample_data(num_samples, dataset=None, seed=0):
    """Same as index_data, but only samples num_samples from the full dataset."""
    index = index_by_label(dataset)
    # we want the sampling to be deterministic and inclusive of previous samples
    random.seed(seed)
    sampled_index = []
    while len(sampled_index) < num_samples:
        choice = random.choice(index.items())
        if choice not in sampled_index:
            sampled_index.append(choice)
    return index_data(sampled_index)


if SAMPLE:
    database_paths, database_labels, query_paths, query_labels = sample_data(SAMPLE)
else:
    database_paths, database_labels, query_paths, query_labels = index_data()


def indices_with_label(target_label, labels):
    """Get indices in labels with target_label."""
    indices = []
    for i, label in enumerate(labels):
        if label == target_label:
            indices.append(i)
    return indices


def load_data(dataset=None):
    """Return an iterator of (label, image) for all images."""
    for label, img_path in index_images(dataset):
        yield label, imread(img_path)


def get_basename_to_path_dict(dataset=None):
    """Generate a dictionary mapping basenames of images to their paths."""
    basename_to_path = {}
    for _, path in index_images(dataset):
        basename = os.path.basename(path)
        basename_to_path[basename] = path
    return basename_to_path
