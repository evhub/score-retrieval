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
    return random.choice(correct_indices) if correct_indices else not_ind

def rep_count(repeat, limit):
    """Yield each number repeat times up to length limit."""
    ind = 0
    i = 0
    done = False
    while not done:
        i += 1
        for _ in range(repeat):
            if ind >= limit:
                done = True
                break
            yield i
            ind += 1

db = {
    "cluster": list(rep_count(CLUSTER_LEN, len(train_images))),
    "qidxs": range(len(train_images)),
    "pidxs": [
        random_index(train_labels, label, i)
        for i, label in enumerate(train_labels)
    ],
}
