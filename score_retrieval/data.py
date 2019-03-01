from __future__ import division

import os
import random
from collections import defaultdict

import cv2

from score_retrieval.constants import (
    get_dataset_dir,
    IMG_EXT,
    DEFAULT_DATASET,
    MAX_QUERIES_PER_LABEL,
    TEST_RATIO,
    TRAIN_RATIO,
)


def get_label(image_path):
    """Get the label for the given image."""
    piece_dir = os.path.dirname(image_path)
    composer_dir = os.path.dirname(piece_dir)
    dataset_dir = os.path.dirname(composer_dir)
    return os.path.relpath(piece_dir, dataset_dir)


def index_images(dataset=None):
    """Return an iterator of (label, path) for all images."""
    data_dir = get_dataset_dir(dataset)
    for dirpath, _, filenames in os.walk(data_dir):
        for fname in filenames:
            if os.path.splitext(fname)[-1] == IMG_EXT:
                img_path = os.path.join(dirpath, fname)
                yield get_label(img_path), img_path


def gen_label_name_index(indexed_images, sort=False):
    """Return dict mapping labels to dict mapping names to image paths."""
    index = defaultdict(lambda: defaultdict(list))
    for label, img_path in indexed_images:
        name, ind = os.path.splitext(os.path.basename(img_path))[0].split("_")
        ind = int(ind)
        group = index[label][name]
        if sort:
            while len(group) <= ind:
                group.append(None)
            group[ind] = img_path
        else:
            group.append(img_path)
    return index


def index_by_label_and_name(dataset=None, sort=False):
    """Run gen_label_name_index on the given dataset."""
    return gen_label_name_index(index_images(dataset), sort=sort)


def index_data(base_index=None, skip_queryless=True, max_queries_per_label=MAX_QUERIES_PER_LABEL):
    """Return database_paths, database_labels, query_paths, query_labels lists."""
    if base_index is None:
        base_index = index_by_label_and_name()
    database_paths = []
    database_labels = []
    query_paths = []
    query_labels = []
    for label, name_index in base_index.items():

        names = tuple(name_index.keys())
        if len(names) < 2:
            if skip_queryless:
                continue
            head_names, tail_names = names, []
        else:
            head_names, tail_names = names[:-1], names[-1:]

        if max_queries_per_label:
            tail_names = tail_names[max_queries_per_label:]

        head_paths = []
        for name in head_names:
            head_paths.extend(name_index[name])

        tail_paths = []
        for name in tail_names:
            tail_paths.extend(name_index[name])

        database_paths.extend(head_paths)
        database_labels.extend([label]*len(head_paths))

        query_paths.extend(tail_paths)
        query_labels.extend([label]*len(tail_paths))

    return database_paths, database_labels, query_paths, query_labels


def get_split_indexes(split_ratios, base_index=None, seed=0):
    """Splits the given index into the given portions."""
    if base_index is None:
        base_index = index_by_label_and_name()

    # generate cumulative ratios
    cum_split_ratios = []
    cum_ratio = 0
    for ratio in split_ratios:
        cum_split_ratios.append(cum_ratio + ratio)
        cum_ratio += ratio

    # deterministically shuffle index
    random.seed(seed)
    shuffled_index = random.sample(base_index.keys(), len(base_index))

    # split index
    split_indexes = [{} for _ in range(len(cum_split_ratios))]
    for i, k in enumerate(shuffled_index):
        v = base_index[k]
        ratio_thru = i/len(base_index)
        for j, split_ratio in enumerate(cum_split_ratios):
            if ratio_thru <= split_ratio:
                split_indexes[j][k] = v
                break
    return split_indexes


def deindex(base_index):
    """Convert a label name index into paths, labels."""
    paths = []
    labels = []
    for label, name_index in base_index.items():
        for name, name_paths in name_index.items():
            for img_path in name_paths:
                paths.append(img_path)
                labels.append(label)
    return paths, labels


test_label_name_index, train_label_name_index = get_split_indexes([
    TEST_RATIO,
    TRAIN_RATIO,
])
database_paths, database_labels, query_paths, query_labels = index_data(test_label_name_index)
train_paths, train_labels = deindex(train_label_name_index)


def indices_with_label(target_label, labels):
    """Get indices in labels with target_label."""
    indices = []
    for i, label in enumerate(labels):
        if label == target_label:
            indices.append(i)
    return indices


def load_img(img_path, grayscale=True):
    """Load an image."""
    cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def load_data(dataset=None, grayscale=True):
    """Return an iterator of (label, image) for all images."""
    for label, img_path in index_images(dataset):
        yield label, load_img(img_path, grayscale=grayscale)


def get_basename_to_path_dict(dataset=None):
    """Generate a dictionary mapping basenames of images to their paths."""
    basename_to_path = {}
    for _, path in index_images(dataset):
        basename = os.path.basename(path)
        basename_to_path[basename] = path
    return basename_to_path
