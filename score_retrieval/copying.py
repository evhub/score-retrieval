import os
import random
import shutil
import traceback
import re
import sys

from score_retrieval.constants import (
    SCRAPE_DIR,
    DATA_DIR,
    SEARCH_HTML_FOR,
    HTML_FNAME,
    SORT_HTML_BY,
)

if sys.version_info < (3,):
    input = raw_input


html_sort_dict = {}


def index_pieces(num_pieces):
    """Index all the piece directories in the scrape directory."""
    got_pieces = 0
    num_missing_html = 0
    num_missing_regex = 0
    complete_walk = list(os.walk(SCRAPE_DIR))
    random.shuffle(complete_walk)
    for dirpath, _, filenames in complete_walk:
        for fname in filenames:
            if os.path.splitext(fname)[-1] == ".pdf":
                if SEARCH_HTML_FOR is not None:
                    html_path = os.path.join(dirpath, HTML_FNAME)
                    if not os.path.exists(html_path):
                        num_missing_html += 1
                        break
                    with open(html_path, "r") as html_file:
                        html = html_file.read()
                        if not SEARCH_HTML_FOR.search(html):
                            num_missing_regex += 1
                            break
                        if SORT_HTML_BY is not None:
                            match = SORT_HTML_BY.search(html)
                            if not match:
                                print("No HTML sort tag in {}.".format(fname))
                                continue
                            try:
                                html_sort_dict[dirpath] = float(match[0])
                            except (TypeError, ValueError):
                                print("Bad HTML sort tag in {}.".format(fname))
                                continue
                got_pieces += 1
                yield dirpath
                break
        if got_pieces >= num_pieces:
            break
    print("Indexed {} pieces ({} had no HTML; {} were missing desired regex in their HTML).".format(got_pieces, num_missing_html, num_missing_regex))


def copy_data(dataset_name, num_pieces):
    """Copy num_pieces worth of data to the data directory."""
    data_dir = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    print("Copying {} pieces...".format(num_pieces))

    if SORT_HTML_BY is None:
        index = index_pieces(num_pieces)
    else:
        index = list(index_pieces(float("inf")))
        index.sort(key=lambda dirpath: html_sort_dict[dirpath], reverse=True)
        index = index[num_pieces:]

    for dirpath in index:
        relpath = os.path.relpath(dirpath, SCRAPE_DIR)
        newpath = os.path.join(data_dir, relpath)

        print("Saving: {} -> {}".format(dirpath, newpath))
        try:
            shutil.copytree(dirpath, newpath)
        except Exception:
            traceback.print_exc()
            print("Skipping: {} -> {}".format(dirpath, newpath))


if __name__ == "__main__":
    dataset_name = input("enter name of new dataset: ")
    num_pieces = int(input("enter number of pieces to copy: "))
    copy_data(dataset_name, num_pieces)
