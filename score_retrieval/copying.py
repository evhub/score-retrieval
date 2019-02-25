import os
import random
import shutil
import traceback
import re
import sys

from score_retrieval.constants import (
    SCRAPE_DIR,
    DATA_DIR,
    HTML_DIR,
    SEARCH_HTML_FOR,
    HTML_EXT,
)

if sys.version_info < (3,):
    input = raw_input


def index_all_pieces():
    """Index all the piece directories in the scrape directory."""
    all_pieces = []
    num_no_html = 0
    num_bad_html = 0
    for dirpath, _, filenames in os.walk(SCRAPE_DIR):
        for fname in filenames:
            if os.path.splitext(fname)[-1] == ".pdf":
                if SEARCH_HTML_FOR:
                    name = os.path.splitext(fname)[0]
                    html_path = os.path.join(HTML_DIR, name + HTML_EXT)
                    if not os.path.exists(html_path):
                        num_no_html += 1
                        continue
                    with open(html_path, "r") as html_file:
                        html = html_file.read()
                        if not SEARCH_HTML_FOR.search(html):
                            num_bad_html += 1
                            break
                all_pieces.append(dirpath)
                break
    print("Indexed {} pieces ({} had no HTML; {} were missing desired regex in their HTML).")
    return all_pieces



def copy_data(dataset_name, num_pieces):
    """Copy num_pieces worth of data to the data directory."""
    data_dir = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print("Copying {} pieces...".format(num_pieces))
    all_pieces = index_all_pieces()
    for dirpath in random.sample(all_pieces, num_pieces):
        relpath = os.path.relpath(dirpath, SCRAPE_DIR)
        newpath = os.path.join(data_dir, relpath)
        print("Saving: {} -> {}".format(dirpath, newpath))
        try:
            shutil.copytree(dirpath, newpath)
        except shutil.Error:
            traceback.print_exc()
            print("Skipping: {} -> {}".format(dirpath, newpath))


if __name__ == "__main__":
    dataset_name = input("enter name of new dataset: ")
    num_pieces = int(input("enter number of pieces to attempt to copy: "))
    copy_data(dataset_name, num_pieces)
