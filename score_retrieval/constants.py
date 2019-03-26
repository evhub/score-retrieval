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
SORT_HTML_BY = re.compile(r"<span class='current-rating' id='current-rating-\d+' style='width:0%;'>([\d\.]+)/10</span>", flags=re.U)

# data migration constants
IMG_EXT = ".png"
DPI = 50

# data renaming constants
BASE_DATASET = "top50_piano_dataset"
QUERY_NAME = "query"
DB_NAME = "db"

# universal dataset constants
EXPORT_TEST_AS_TRAIN = False
START_PAGE = 1
IGNORE_IMAGES = None

# single dataset constants
DEFAULT_DATASET = "piano_dataset"
MAX_QUERIES_PER_LABEL = None
TEST_RATIO = 0.95
TRAIN_RATIO = 0.05
TRAIN_ON_EXCESS = True

# multi dataset constants
USE_MULTIDATASET = True
QUERY_DATASET = "query_dataset"
DB_DATASET = "db_dataset"
AUGMENT_DB_DATASET = "new_piano_dataset"
TRAIN_DATASET = "new_piano_dataset"
AUGMENT_DB_TO = 6000
MULTIDATASET_QUERY_RATIO = 1.0
MULTIDATASET_DB_RATIO = 1.0
MULTIDATASET_TRAIN_RATIO = 1.0
ALLOWED_AUGMENT_COMPOSERS = (
    "Wagner,_Richard",
    "Haydn,_Joseph",
    "Handel,_George_Frideric",
    "Dvo%C5%99%C3%A1k,_Anton%C3%ADn",
    "Stravinsky,_Igor",
    "Verdi,_Giuseppe",
    "Mahler,_Gustav",
    "Prokofiev,_Sergey",
    "Berlioz,_Hector",
    "Puccini,_Giacomo",
    "Palestrina,_Giovanni_Pierluigi_da",
    "Bruckner,_Anton",
    "Telemann,_Georg_Philipp",
    "Sibelius,_Jean",
    "Rossini,_Gioacchino",
    "Gluck,_Christoph_Willibald",
    "Hindemith,_Paul",
    "Monteverdi,_Claudio",
    "Franco,_Cesare",
    "Bizet,_Georges",
    "Rameau,_Jean-Philippe",
    "Faur%C3%A9,_Gabriel",
    "Rimsky-Korsakov,_Nikolay",
    "Donizetti,_Gaetano",
    "Smetana,_Bed%C5%99ich",
    "Jan%C3%A1%C4%8Dek,_Leo%C5%A1",
    "Couperin,_Fran%C3%A7ois",
)

# cli arg processing
arguments = argparse.ArgumentParser(
    prog="score-retrieval",
)
arguments.add_argument(
    "--dataset",
    metavar="name",
    type=str,
    default=DEFAULT_DATASET,
    help="defaults to {}".format(DEFAULT_DATASET)
)
arguments.add_argument(
    "--test-ratio",
    metavar="ratio",
    type=float,
    default=TEST_RATIO,
    help="defaults to {}".format(TEST_RATIO)
)
arguments.add_argument(
    "--train-ratio",
    metavar="ratio",
    type=float,
    default=TRAIN_RATIO,
    help="defaults to {}".format(TRAIN_RATIO)
)
arguments.add_argument(
    "--train-on-excess",
    metavar="bool",
    type=bool,
    default=TRAIN_ON_EXCESS,
    help="defaults to {}".format(TRAIN_ON_EXCESS)
)

# vector saving constants
ALG = "bar splitting"

# retrieval constants
LIN_WEIGHT = 0.0
SLOPE_WEIGHT = 0.25

# evaluation constants
TOP_N_ACCURACY = 5

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
