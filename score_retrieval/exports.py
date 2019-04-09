import random

from score_retrieval.constants import (
    CLUSTER_LEN,
    EXPORT_TEST_AS_TRAIN,
)
from score_retrieval.data import (
    database_paths as images,
    database_labels as image_labels,
    query_paths as qimages,
    query_labels as qimage_labels,
    indices_with_label,
    get_label_set,
)


# testing exports
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


# training exports
if EXPORT_TEST_AS_TRAIN:
    train_images = images + qimages
    train_labels = image_labels + qimage_labels
else:
    from score_retrieval.data import (
    train_paths as train_images,
    train_labels,
)

def random_index(label_list, label, not_ind):
    """Choose random index from labels with the given label."""
    correct_indices = []
    for i, test_label in enumerate(label_list):
        if test_label == label and i != not_ind:
            correct_indices.append(i)
    if correct_indices:
        return random.choice(correct_indices)
    else:
        print("Only one image with label {} in training dataset.".format(label))
        return not_ind

train_label_set = get_label_set(train_labels)

db = {
    "cluster": [train_label_set.index(label) for label in train_labels],
    "qidxs": range(len(train_images)),
    "pidxs": [
        random_index(train_labels, label, i)
        for i, label in enumerate(train_labels)
    ],
}
