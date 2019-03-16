import numpy as np

from score_retrieval.data import (
    query_labels,
    database_labels,
    database_paths,
)


def path_ranking_to_index_ranking(path_ranking, db_paths=database_paths):
    """Convert ranking of paths into ranking of indices."""
    query_ranking = []
    for database_path in path_ranking:
        database_index = db_paths.index(database_path)
        query_ranking.append(database_index)
    return query_ranking


def get_pos_ranks(query_rankings, db_labels=database_labels):
    """
    query_rankings[i][j] = index in db_labels of the
        (j+1)th ranked database image for the (i+1)th query

    returns: generator of lists of rankings starting
        from 0 of positive labels for each query
    """
    for query_index, query_label in enumerate(query_labels):
        # first rank all the labels
        ranked_labels = []
        for database_index in enumerate(query_rankings[query_index]):
            label = db_labels[database_index]
            if label not in ranked_labels:
                ranked_labels.append(label)

        # then yield an array of just the rank of the correct label
        pos_rank = ranked_labels.index(query_label)
        yield np.array([pos_rank])


def compute_mrr(query_rankings, db_labels=database_labels):
    """Compute MRR for query_rankings as specified in get_pos_ranks."""
    mrrs = []
    for pos_ranks in get_pos_ranks(query_rankings, db_labels):
        if len(pos_ranks):
            mrrs.append(np.mean(individual_mrr(pos_ranks)))
    return np.mean(np.array(mrrs))


def individual_mrr(pos_rank):
    """Compute a single MRR from the given pos_ranks."""
    return 1/(pos_rank + 1)


def compute_acc(query_rankings, top_n=1, db_labels=database_labels):
    """Compute accuracy for query_rankings as specified in get_pos_ranks."""
    total = 0.0
    correct = 0.0
    for pos_ranks in get_pos_ranks(query_rankings, db_labels):
        if len(pos_ranks):
            total += 1
            for i in range(top_n):
                if i in pos_ranks:
                    correct += 1
                    break
    return correct / total
