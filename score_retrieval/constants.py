import os
import re
import random
import argparse


random.seed(0)

# base data constants
DATA_DIR = "/home/ehubinger/score-retrieval/data"
if not os.path.exists(DATA_DIR):
    print("Could not find data dir {}, defaulting to local data dir.".format(DATA_DIR))
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# data copying constants
SCRAPE_DIR = "/data1/dbashir/Project/score_scrape/results/composer"
HTML_FNAME = "html.txt"
SEARCH_HTML_FOR = re.compile(r"<th>Instrumentation\n</th>\n<td>\s*Piano\s*\n</td>", flags=re.U)

# data migration constants
IMG_EXT = ".png"
DPI = 50
START_PAGE = None
END_PAGE = None

# dataset constants
DEFAULT_DATASET = "piano_dataset"
MAX_QUERIES_PER_LABEL = None
TEST_RATIO = 0.95
TRAIN_RATIO = 0.05
TRAIN_ON_EXCESS = True
EXPORT_TEST_AS_TRAIN = False

# cli arg processing
arguments = argparse.ArgumentParser(
    prog="score-retrieval",
)
arguments.add_argument(
    "--dataset",
    metavar="name",
    type=str,
    default=DEFAULT_DATASET,
)
arguments.add_argument(
    "--test-ratio",
    metavar="ratio",
    type=float,
    default=TEST_RATIO,
)
arguments.add_argument(
    "--train-ratio",
    metavar="ratio",
    type=float,
    default=TRAIN_RATIO,
)
arguments.add_argument(
    "--train-on-excess",
    metavar="bool",
    type=bool,
    default=TRAIN_ON_EXCESS,
)

# retrieval constants
LIN_WEIGHT = 0.0
SLOPE_WEIGHT = 0.25

# exporting constants
CLUSTER_LEN = 64

# utilities
def get_dataset_dir(dataset=None):
    """Get the directory for the given dataset."""
    data_dir = DATA_DIR
    if dataset is None:
        dataset = DEFAULT_DATASET
    if dataset:
        data_dir = os.path.join(data_dir, dataset)
    return data_dir
