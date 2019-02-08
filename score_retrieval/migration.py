import os
import traceback

from pdf2image import convert_from_path

from score_retrieval.constants import (
    DATA_DIR,
    IMG_EXT,
    DPI,
)


def save_page(pdf_path, img_path):
    """Save middle page of given pdf as an image."""
    try:
        pages = convert_from_path(pdf_path, DPI)
    except Exception:
        traceback.print_exc()
        print("Failed to save {} -> {}.".format(pdf_path, img_path))
    else:
        page_ind = len(pages)//2
        pages[page_ind].save(img_path, os.path.splitext(img_path)[-1].lstrip("."))


def migrate_pdfs():
    """Migrate all pdfs to images."""
    print("Migrating {}...".format(DATA_DIR))
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for pdf_file in filenames:
            name, ext = os.path.splitext(pdf_file)
            if ext == ".pdf":
                img_file = name + IMG_EXT
                if img_file in filenames:
                    print("Skipping {}...".format(img_file))
                else:
                    print("Saving {}...".format(img_file))
                    pdf_path = os.path.join(dirpath, pdf_file)
                    img_path = os.path.join(dirpath, img_file)
                    save_page(pdf_path, img_path)
                    print("Saved {}.".format(img_file))


if __name__ == "__main__":
    migrate_pdfs()
