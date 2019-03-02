from score_retrieval.data import (
    database_paths as images,
    database_labels as image_labels,
    query_paths as qimages,
    query_labels as qimage_labels,
    train_paths as train_images,
    train_labels,
    indices_with_label,
)


gnd = [
    {
        "ok": indices_with_label(label, image_labels),
        "junk": [],
    } for label in qimage_labels
]

cfg = {
    "gnd": gnd,
    "n": len(images),
    "im_fname": lambda self, i: images[i],
    "nq": len(qimages),
    "qim_fname": lambda self, i: qimages[i],
}

bbxs = None

db = {
    "cluster": None,
    "qidxs": None,
    "pidxs": None,
}
