import os
import re


SCRAPE_DIR = "/data1/dbashir/Project/score_scrape/results/composer"
HTML_DIR = "/data1/dbashir/Project/results"
HTML_EXT = ".txt"
SEARCH_HTML_FOR = re.compile(r"<th>Instrumentation\n</th>\n<td>.*Piano.*\n</td>", flags=re.U)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

IMG_EXT = ".png"

DPI = 50

MAX_QUERIES_PER_LABEL = 5

# size of the subset of the data to sample
SAMPLE = False

VECTOR_LEN = 128

DEFAULT_DATASET = "mini_dataset"

START_PAGE = None
END_PAGE = None


def get_dataset_dir(dataset=None):
    """Get the directory for the given dataset."""
    data_dir = DATA_DIR
    if dataset is None:
        dataset = DEFAULT_DATASET
    if dataset:
        data_dir = os.path.join(data_dir, dataset)
    return data_dir
