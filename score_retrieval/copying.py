import os
import random
import shutil
import traceback

from score_retrieval.constants import (
    SCRAPE_DIR,
    DATA_DIR,
)


def index_all_pieces():
    """Index all the piece directories in the scrape directory."""
    for dirpath, _, filenames in os.walk(SCRAPE_DIR):
        for fname in filenames:
            if os.path.splitext(fname)[-1] == ".pdf":
                yield dirpath
                break


def copy_data(num_pieces):
    """Copy num_pieces worth of data to the data directory."""
    print("Copying {} pieces...".format(num_pieces))
    all_pieces = list(index_all_pieces())
    for dirpath in random.sample(all_pieces, num_pieces):
        relpath = os.path.relpath(dirpath, SCRAPE_DIR)
        newpath = os.path.join(DATA_DIR, relpath)
        print("Saving: {} -> {}".format(dirpath, newpath))
        try:
            shutil.copytree(dirpath, newpath)
        except shutil.Error:
            traceback.print_exc()
            print("Skipping: {} -> {}".format(dirpath, newpath))


if __name__ == "__main__":
    copy_data(int(input("enter number of pieces to copy: ")))
