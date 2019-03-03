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
    train_label_name_index,
    index_data,
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

if EXPORT_TEST_AS_TRAIN:
    train_db_images, train_db_labels, train_query_images, train_query_labels = images, image_labels, qimages, qimage_labels
else:
    train_db_images, train_db_labels, train_query_images, train_query_labels = index_data(train_label_name_index)

train_images = train_query_images + train_db_images

def random_index(label_list, label):
    """Choose random index from labels with the given label."""
    correct_indices = []
    for i, test_label in enumerate(label_list):
        if test_label == label:
            correct_indices.append(i)
    return random.choice(correct_indices)

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
    "qidxs": range(len(train_query_images)),
    "pidxs": [
        len(train_query_images)
        + random_index(train_db_labels, label)
        for i, label in enumerate(train_query_labels)
    ],
}
