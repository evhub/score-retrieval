import os
import shutil
from zlib import crc32

from score_retrieval.constants import (
    BASE_DATASET,
    QUERY_NAME,
    DB_NAME,
    QUERY_DATASET,
    DB_DATASET,
    DATA_DIR,
)
from score_retrieval.data import get_dataset_dir


def checksum(data):
    """Compute a checksum of the given data."""
    return hex(crc32(data) & 0xffffffff)


def rename(base_dataset=BASE_DATASET, query_dataset=QUERY_DATASET, db_dataset=DB_DATASET, query_name=QUERY_NAME, db_name=DB_NAME):
    base_dir = get_dataset_dir(base_dataset)
    query_dir = get_dataset_dir(query_dataset)
    db_dir = get_dataset_dir(db_dataset)
    for dirpath, _, filenames in os.walk(base_dir):
        for fname in filenames:
            name, ext = os.path.splitext(fname)
            if ext == ".pdf":
                base_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(base_path, base_dir)
                rel_dir = os.path.dirname(rel_path)
                new_fname = checksum(rel_path) + ".pdf"
                if name == query_name:
                    new_path = os.path.join(query_dir, rel_dir, new_fname)
                elif name == db_name:
                    new_path = os.path.join(db_dir, rel_dir, new_fname)
                else:
                    raise ValueError("got unknown filename {} (must be {}.pdf or {}.pdf for renaming)".format(fname, query_name, db_name))
                print("Copying {} -> {}...".format(base_path, new_path))
                shutil.copy(base_path, new_path)


if __name__ == "__main__":
    rename()
