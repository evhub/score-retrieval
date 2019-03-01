import os
import random
from collections import defaultdict

from scipy.ndimage import imread

from score_retrieval.constants import (
    get_dataset_dir,
    IMG_EXT,
    SAMPLE,
    DEFAULT_DATASET,
    MAX_QUERIES_PER_LABEL,
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


def index_data(base_index=None, dataset=None, skip_queryless=True, max_queries_per_label=MAX_QUERIES_PER_LABEL):
    """Return database_paths, database_labels, query_paths, query_labels lists."""
    if base_index is None:
        base_index = index_by_label_and_name(dataset)
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


def sample_data(num_samples, dataset=None, seed=0):
    """Same as index_data, but only samples num_samples from the full dataset."""
    index = index_by_label_and_name(dataset)
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
