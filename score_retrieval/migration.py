import os
import traceback

from pdf2image import convert_from_path

from score_retrieval.constants import (
    arguments,
    get_dataset_dir,
    IMG_EXT,
    DPI,
)


def save_pages(pdf_path, name, save_dir):
    """Save all page of given pdf as image."""
    try:
        pages = convert_from_path(pdf_path, DPI)
    except Exception:
        traceback.print_exc()
        print("Failed to save {} -> {}/{}.".format(pdf_path, save_dir, name))
    else:
        for i, page in enumerate(pages):
            page_path = os.path.join(save_dir, name) + "_" + str(i) + IMG_EXT
            if os.path.exists(page_path):
                print("Skipping {}...".format(page_path))
            else:
                print("Saving {}...".format(page_path))
                page.save(page_path, os.path.splitext(page_path)[-1].lstrip("."))
                print("Saved {}.".format(page_path))


def migrate_pdfs(dataset=None):
    """Migrate all pdfs to images."""
    data_dir = get_dataset_dir(dataset)
    print("Migrating {}...".format(data_dir))
    for dirpath, _, filenames in os.walk(data_dir):
        for pdf_file in filenames:
            name, ext = os.path.splitext(pdf_file)
            if ext == ".pdf":
                pdf_path = os.path.join(dirpath, pdf_file)
                save_pages(pdf_path, name, dirpath)


if __name__ == "__main__":
    migrate_pdfs(arguments.parse_args().dataset)
