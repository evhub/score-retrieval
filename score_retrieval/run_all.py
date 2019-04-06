from score_retrieval.constants import arguments
from score_retrieval.data import gen_data_from_args
from score_retrieval.vec_db import generate_vectors_from_args
from score_retrieval.retrieval import run_retrieval


def main():
    """Do vector generation and retrieval."""
    parsed_args = arguments.parse_args()

    # generate vectors
    generate_vectors_from_args(parsed_args)

    # run retrieval
    _data = gen_data_from_args(parsed_args)
    run_retrieval(query_paths=_data["query_paths"], database_paths=_data["database_paths"])


if __name__ == "__main__":
    main()
