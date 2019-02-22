import os
import traceback

from pdf2image import convert_from_path

from score_retrieval.constants import (
    DATA_DIR,
    IMG_EXT,
    DPI,
    START_PAGE,
    END_PAGE,
)


def save_pages(pdf_path, name, save_dir):
    """Save all page of given pdf as image."""
    try:
        pages = convert_from_path(pdf_path, DPI)
    except Exception:
        traceback.print_exc()
        print("Failed to save {} -> {}/{}.".format(pdf_path, save_dir, name))
    else:

        if START_PAGE and END_PAGE:
            pages = pages[START_PAGE:END_PAGE]
        elif START_PAGE:
            pages = pages[START_PAGE:]
        elif END_PAGE:
            pages = pages[:END_PAGE]

        for i, page in enumerate(pages):
            page_path = os.path.join(save_dir, name) + "_" + str(i) + IMG_EXT
            if os.path.exists(page_path):
                print("Skipping {}...".format(page_path))
            else:
                print("Saving {}...".format(page_path))
                page.save(page_path, os.path.splitext(page_path)[-1].lstrip("."))
                print("Saved {}.".format(page_path))


def migrate_pdfs():
    """Migrate all pdfs to images."""
    print("Migrating {}...".format(DATA_DIR))
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for pdf_file in filenames:
            name, ext = os.path.splitext(pdf_file)
            if ext == ".pdf":
                pdf_path = os.path.join(dirpath, pdf_file)
                save_pages(pdf_path, name, dirpath)


if __name__ == "__main__":
    migrate_pdfs()
