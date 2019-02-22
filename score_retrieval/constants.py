import os


SCRAPE_DIR = "/data1/dbashir/Project/score_scrape/results/composer"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

IMG_EXT = ".png"

DPI = 50

# size of the subset of the data to sample
SAMPLE = False

VECTOR_LEN = 128

DEFAULT_DATASET = "mini_dataset"

START_PAGE = None

END_PAGE = None
