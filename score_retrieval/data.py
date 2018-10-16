import os

from pdf2image import convert_from_path
from scipy.ndimage import imread

from score_retrieval.constants import (
    DATA_DIR,
)


def save_first_page(pdf_path, img_path, dpi=300):
    """Save first page of given pdf as an image."""
    pages = convert_from_path(pdf_path, dpi)
    pages[-1].save(img_path, os.path.splitext(img_path)[-1])


def migrate_pdfs(img_ext=".png", dpi=300):
    """Migrate all pdfs to images."""
    print("Migrating {}...".format(DATA_DIR))
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for pdf_file in filenames:
            name, ext = os.path.splitext(pdf_file)
            if ext == ".pdf":
                img_file = name + img_ext
                if img_file in filenames:
                    print("Skipping {}...".format(img_file))
                else:
                    print("Saving {}...".format(img_file))
                    pdf_path = os.path.join(dirpath, pdf_file)
                    img_path = os.path.join(dirpath, img_file)
                    save_first_page(pdf_path, img_path, dpi)
                    print("Saved {}.".format(img_file))


def load_imgs(img_ext=".png"):
    """Load all images in the dataset as an iterator."""
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for fname in filenames:
            if os.path.splitext(fname)[-1] == img_ext:
                img_path = os.path.join(dirpath, fname)
                yield imread(img_path)


if __name__ == "__main__":
    migrate_pdfs()
