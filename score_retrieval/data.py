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
    START_PAGE,
    IGNORE_IMAGES,
    USE_MULTIDATASET,
    QUERY_DATASET,
    DB_DATASET,
    AUGMENT_DB_DATASET,
    TRAIN_DATASET,
    AUGMENT_DB_TO,
    MULTIDATASET_QUERY_RATIO,
    MULTIDATASET_DB_RATIO,
    MULTIDATASET_TRAIN_RATIO,
    ALLOWED_AUGMENT_TRAIN_COMPOSERS,
    arguments,
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


def get_composer(image_path):
    piece_dir = os.path.dirname(image_path)
    composer_dir = os.path.dirname(piece_dir)
    return os.path.basename(composer_dir)


def get_label(image_path):
    """Get the label for the given image."""
    rel_path = os.path.relpath(image_path, DATA_DIR)
    dataset_dir = os.path.join(DATA_DIR, top_dir(rel_path))
    piece_dir = os.path.dirname(image_path)
    return os.path.relpath(piece_dir, dataset_dir)


def get_name_ind(img_path):
    """Get the name and index of the given image."""
    name, ind = os.path.splitext(os.path.basename(img_path))[0].split("_")
    return name, int(ind)


def index_images(dataset=None):
    """Return an iterator of (label, path) for all images."""
    data_dir = get_dataset_dir(dataset)
    for dirpath, _, filenames in os.walk(data_dir):
        for fname in filenames:
            if os.path.splitext(fname)[-1] == IMG_EXT:
                if IGNORE_IMAGES and fname in IGNORE_IMAGES:
                    continue
                img_path = os.path.join(dirpath, fname)
                name, ind = get_name_ind(img_path)
                if START_PAGE is not None and ind < START_PAGE:
                    continue
                yield get_label(img_path), img_path


def indices_with_label(target_label, labels):
    """Get indices in labels with target_label."""
    indices = []
    for i, label in enumerate(labels):
        if label == target_label:
            indices.append(i)
    return indices


def load_img(img_path, grayscale=False):
    """Load an image."""
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def load_data(dataset=None, grayscale=False):
    """Return an iterator of (label, image) for all images."""
    for label, img_path in index_images(dataset):
        yield label, load_img(img_path, grayscale=grayscale)


def get_basename_to_path_dict(dataset=None):
    """Generate a dictionary mapping basenames of images to their paths."""
    basename_to_path = {}
    for _, path in index_images(dataset):
        basename = os.path.basename(path)
        if basename in basename_to_path:
            print("Found duplicate image index: {}".format(basename))
        basename_to_path[basename] = path
    return basename_to_path


def gen_label_name_index(indexed_images, sort=False):
    """Return dict mapping labels to dict mapping names to image paths."""
    index = defaultdict(lambda: defaultdict(list))
    for label, img_path in indexed_images:
        name, ind = get_name_ind(img_path)
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
        if len(names) >= 2:
            head_names, tail_names = names[:1], names[1:]

            if max_queries_per_label:
                tail_names, excess_names = tail_names[max_queries_per_label:], tail_names[max_queries_per_label:]
            else:
                excess_names = []

        elif skip_queryless:
            head_names, tail_names, excess_names = [], [], names

        else:
            head_names, tail_names, excess_names = names, [], []

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


def deindex(base_index, ignore_labels=None, ignore_names=None, ignore_composers=None, allow_composers=None):
    """Convert a label name index into paths, labels."""
    paths = []
    labels = []
    for label, name_index in base_index.items():
        if ignore_labels and label in ignore_labels:
            continue
        for name, name_paths in name_index.items():
            if ignore_names and name in ignore_names:
                continue
            for img_path in name_paths:
                composer = get_composer(img_path)
                if allow_composers and composer not in allow_composers:
                    continue
                if ignore_composers and composer in ignore_composers:
                    continue
                paths.append(img_path)
                labels.append(label)
    return paths, labels


def get_names(paths):
    """Get a set of all the names in paths."""
    name_set = set()
    for img_path in paths:
        name, ind = get_name_ind(img_path)
        name_set.add(name)
    return name_set


def num_names(paths):
    """Get the number of unique names in paths."""
    return len(get_names(paths))


def get_label_set(db_labels):
    """Gets a list of unique labels in db_labels."""
    return list(set(db_labels))


# train and test data generators
def gen_single_dataset_data(dataset=DEFAULT_DATASET, test_ratio=TEST_RATIO, train_ratio=TRAIN_RATIO, train_on_excess=TRAIN_ON_EXCESS):
    """Generate all database endpoints from the given dataset."""
    datasets = (dataset,)
    base_index = index_by_label_and_name(dataset)

    test_label_name_index, train_label_name_index = get_split_indexes([
        test_ratio,
        train_ratio,
    ], base_index)

    train_paths, train_labels = deindex(train_label_name_index)

    database_paths, database_labels, query_paths, query_labels = index_data(
        test_label_name_index,
        excess_paths=train_paths if train_on_excess else None,
        excess_labels=train_labels if train_on_excess else None,
    )

    return locals()


def gen_multi_dataset_data(
        query_dataset=QUERY_DATASET,
        db_dataset=DB_DATASET,
        train_dataset=TRAIN_DATASET,
        augment_db_dataset=AUGMENT_DB_DATASET,
        augment_db_to=AUGMENT_DB_TO,
        query_ratio=MULTIDATASET_QUERY_RATIO,
        db_ratio=MULTIDATASET_DB_RATIO,
        train_ratio=MULTIDATASET_TRAIN_RATIO,
        allowed_augment_train_composers=ALLOWED_AUGMENT_TRAIN_COMPOSERS,
    ):
    """Generate all database endpoints from separate datasets."""
    datasets = (
        query_dataset,
        db_dataset,
        train_dataset,
    )

    # generate query data
    query_label_name_index = index_by_label_and_name(query_dataset)
    if query_ratio is not None:
        query_label_name_index, = get_split_indexes([query_ratio], query_label_name_index)
    query_paths, query_labels = deindex(query_label_name_index)

    # generate db data
    db_label_name_index = index_by_label_and_name(db_dataset)
    if db_ratio is not None:
        db_label_name_index, = get_split_indexes([db_ratio], db_label_name_index)
    db_paths, db_labels = deindex(db_label_name_index)

    # augment db data
    if augment_db_dataset is not None:
        assert augment_db_to is not None, "must pass augment_db_to when passing augment_db_dataset"
        augment_db_label_name_index = index_by_label_and_name(augment_db_dataset)
        augment_db_paths, augment_db_labels = deindex(augment_db_label_name_index, allow_composers=allowed_augment_train_composers)
        for path, label in zip(augment_db_paths, augment_db_labels):
            if len(db_paths) >= augment_db_to:
                break
            db_paths.append(path)
            db_labels.append(label)

    # only get db names when db data is finalized
    db_names = get_names(db_paths)

    # generate train data
    train_label_name_index = index_by_label_and_name(train_dataset)
    if train_ratio is not None:
        train_label_name_index, = get_split_indexes([train_ratio], train_label_name_index)
    train_paths, train_labels = deindex(train_label_name_index, ignore_names=db_names, allow_composers=allowed_augment_train_composers)

    return locals()


def gen_data_from_args(parsed_args=None):
    if parsed_args is None:
        parsed_args = arguments.parse_args()
    print("parsed_args =", parsed_args)
    return gen_single_dataset_data(parsed_args.dataset, parsed_args.test_ratio, parsed_args.train_ratio, parsed_args.train_on_excess)


# generate data
if USE_MULTIDATASET:
    _data = gen_multi_dataset_data()
else:
    if __name__ == "__main__":
        _data = gen_data_from_args()
    else:
        _data = gen_single_dataset_data()

datasets = _data["datasets"]
datasets_str = "_".join(datasets)
train_paths = _data["train_paths"]
train_labels = _data["train_labels"]
database_paths = _data["database_paths"]
database_labels = _data["database_labels"]
query_paths = _data["query_paths"]
query_labels = _data["query_labels"]


# display lengths when run directly
if __name__ == "__main__":
    print("datasets: {}".format(datasets_str))
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
