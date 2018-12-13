import os
from collections import defaultdict

from pdf2image import convert_from_path
from scipy.ndimage import imread

from score_retrieval.constants import (
    DATA_DIR,
    IMG_EXT,
    DPI,
)


def save_first_page(pdf_path, img_path):
    """Save first page of given pdf as an image."""
    pages = convert_from_path(pdf_path, DPI)
    pages[-1].save(img_path, os.path.splitext(img_path)[-1].lstrip("."))


def migrate_pdfs():
    """Migrate all pdfs to images."""
    print("Migrating {}...".format(DATA_DIR))
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for pdf_file in filenames:
            name, ext = os.path.splitext(pdf_file)
            if ext == ".pdf":
                img_file = name + IMG_EXT
                if img_file in filenames:
                    print("Skipping {}...".format(img_file))
                else:
                    print("Saving {}...".format(img_file))
                    pdf_path = os.path.join(dirpath, pdf_file)
                    img_path = os.path.join(dirpath, img_file)
                    save_first_page(pdf_path, img_path)
                    print("Saved {}.".format(img_file))


def index_images():
    """Return an iterator of (label, path) for all images."""
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for fname in filenames:
            if os.path.splitext(fname)[-1] == IMG_EXT:
                yield dirpath, os.path.join(dirpath, fname)


def index_by_label():
    """Return dictionary mapping labels to image paths."""
    index = defaultdict(list)
    for label, img_path in index_images():
        index[label].append(img_path)
    return index


def index_data():
    """Return database_paths, database_labels, query_paths, query_labels lists."""
    database_paths = []
    database_labels = []
    query_paths = []
    query_labels = []
    for label, img_paths in index_by_label().items():
        head, tail = img_paths[:-1], img_paths[-1]
        database_paths += head
        database_labels += [label]*len(head)
        query_paths.append(tail)
        query_labels.append(label)
    return database_paths, database_labels, query_paths, query_labels


database_paths, database_labels, query_paths, query_labels = index_data()


def indices_with_label(target_label, labels):
    """Get indices in labels with target_label."""
    indices = []
    for i, label in enumerate(labels):
        if label == target_label:
            indices.append(i)
    return indices


def load_data():
    """Return an iterator of (label, image) for all images."""
    for label, img_path in index_images():
        yield label, imread(img_path)


def get_basename_to_path_dict():
    """Generate a dictionary mapping basenames of images to their paths."""
    basename_to_path = {}
    for _, path in index_images():
        basename = os.path.basename(path)
        basename_to_path[basename] = path
    return basename_to_path


if __name__ == "__main__":
    migrate_pdfs()
