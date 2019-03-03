import os
import re
import random


random.seed(0)

DATA_DIR = "/home/ehubinger/score-retrieval/data"

# data copying constants
SCRAPE_DIR = "/data1/dbashir/Project/score_scrape/results/composer"
HTML_DIR = "/data1/dbashir/Project/results"
HTML_EXT = ".txt"
SEARCH_HTML_FOR = re.compile(r"<th>Instrumentation\n</th>\n<td>.*Piano.*\n</td>", flags=re.U)

# data migration constants
IMG_EXT = ".png"
DPI = 50
START_PAGE = None
END_PAGE = None

# dataset constants
DEFAULT_DATASET = "mini_dataset"  # "piano_dataset"
MAX_QUERIES_PER_LABEL = None  # 1
TEST_RATIO = 1.0  # 0.15
TRAIN_RATIO = 0.05
TRAIN_ON_EXCESS = False
EXPORT_TEST_DATA = True  # False

# retrieval constants
VECTOR_LEN = 128

# exporting constants
CLUSTER_LEN = 78


def get_dataset_dir(dataset=None):
    """Get the directory for the given dataset."""
    data_dir = DATA_DIR
    if dataset is None:
        dataset = DEFAULT_DATASET
    if dataset:
        data_dir = os.path.join(data_dir, dataset)
    return data_dir
