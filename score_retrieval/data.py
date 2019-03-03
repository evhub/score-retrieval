from __future__ import division

import os
import sys
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
    DATA_DIR,
    TRAIN_ON_EXCESS,
)


def top_dir(path):
    """Get the top-level directory of the given path."""
    while True:
        head, tail = os.path.split(path)
        if not head:
            return tail
        elif not tail:
            return head
        else:
            path = head


def get_label(image_path):
    """Get the label for the given image."""
    rel_path = os.path.relpath(image_path, DATA_DIR)
    dataset_dir = os.path.join(DATA_DIR, top_dir(rel_path))
    piece_dir = os.path.dirname(image_path)
    return os.path.relpath(piece_dir, dataset_dir)


def index_images(dataset=None):
    """Return an iterator of (label, path) for all images."""
    data_dir = get_dataset_dir(dataset)
    for dirpath, _, filenames in os.walk(data_dir):
        for fname in filenames:
            if os.path.splitext(fname)[-1] == IMG_EXT:
                img_path = os.path.join(dirpath, fname)
                yield get_label(img_path), img_path


def indices_with_label(target_label, labels):
    """Get indices in labels with target_label."""
    indices = []
    for i, label in enumerate(labels):
        if label == target_label:
            indices.append(i)
    return indices


def load_img(img_path, grayscale=True):
    """Load an image."""
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


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


def get_name_ind(img_path):
    """Get the name and index of the given image."""
    name, ind = os.path.splitext(os.path.basename(img_path))[0].split("_")
    return name, ind


def gen_label_name_index(indexed_images, sort=False):
    """Return dict mapping labels to dict mapping names to image paths."""
    index = defaultdict(lambda: defaultdict(list))
    for label, img_path in indexed_images:
        name, ind = get_name_ind(img_path)
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


def append_names(names, paths, labels, label, name_index):
    """Append all given names to the given paths, labels."""
    new_paths = []
    for name in names:
        new_paths.extend(name_index[name])
    paths.extend(new_paths)
    labels.extend([label]*len(new_paths))


def index_data(base_index=None, skip_queryless=True, max_queries_per_label=MAX_QUERIES_PER_LABEL, excess_paths=None, excess_labels=None):
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
            head_names, tail_names = names[:1], names[1:]

        if max_queries_per_label:
            tail_names, excess_names = tail_names[max_queries_per_label:], tail_names[max_queries_per_label:]
        else:
            excess_names = []

        append_names(head_names, database_paths, database_labels, label, name_index)
        append_names(tail_names, query_paths, query_labels, label, name_index)
        if excess_paths is not None:
            append_names(excess_names, excess_paths, excess_labels, label, name_index)

    return database_paths, database_labels, query_paths, query_labels


def get_split_indexes(split_ratios, base_index=None):
    """Splits the given index into the given portions."""
    if base_index is None:
        base_index = index_by_label_and_name()

    # generate cumulative ratios
    cum_split_ratios = []
    cum_ratio = 0
    for ratio in split_ratios:
        cum_split_ratios.append(cum_ratio + ratio)
        cum_ratio += ratio
    cum_split_ratios[-1] += sys.float_info.epsilon

    # deterministically shuffle index
    shuffled_index = random.sample(base_index.keys(), len(base_index))

    # split index
    split_indexes = [{} for _ in range(len(cum_split_ratios))]
    for i, k in enumerate(shuffled_index):
        v = base_index[k]
        ratio_thru = i/len(base_index)
        for j, split_ratio in enumerate(cum_split_ratios):
            if ratio_thru < split_ratio:
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


def num_names(paths):
    """Get the number of unique names in paths."""
    name_set = set()
    for img_path in paths:
        name, ind = get_name_ind(img_path)
        name_set.add(name)
    return len(name_set)


# generate train and test data
test_label_name_index, train_label_name_index = get_split_indexes([
    TEST_RATIO,
    TRAIN_RATIO,
])

train_paths, train_labels = deindex(train_label_name_index)

database_paths, database_labels, query_paths, query_labels = index_data(
    test_label_name_index,
    excess_paths=train_paths if TRAIN_ON_EXCESS else None,
    excess_labels=train_labels if TRAIN_ON_EXCESS else None,
)


# display lengths when run directly
if __name__ == "__main__":
    print("dataset: {}".format(DEFAULT_DATASET))
    num_db_names = num_names(database_paths)
    print("database: {} images from {} pdfs".format(len(database_paths), num_db_names))
    num_query_names = num_names(query_paths)
    print("query: {} images from {} pdfs".format(len(query_paths), num_query_names))
    num_train_names = num_names(train_paths)
    print("train: {} images from {} pdfs".format(len(train_paths), num_train_names))
    print("total: {} images from {} pdfs".format(
        len(train_paths) + len(database_paths) + len(query_paths),
        num_db_names + num_query_names + num_train_names,
    ))
