.PHONY: install
install:
	pip install -Ue .

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: delete-images
delete-images:
	find . -name '*.png' -delete

.PHONY: delete-vecs
delete-vecs:
	find . -name '*.npy' -delete

.PHONY: pdfs-to-images
pdfs-to-images:
	python ./score_retrieval/migration.py

.PHONY: copy-data
copy-data:
	python ./score_retrieval/copying.py

.PHONY: save-vecs
save-vecs:
	python ./score_retrieval/vec_db.py

.PHONY: run-retrieval
run-retrieval:
	python ./score_retrieval/retrieval.py

.PHONY: run-all
run-all: save-vecs run-retrieval

.PHONY: clean-run
clean-run: delete-vecs run-all

.PHONY: index-data
index-data:
	python ./score_retrieval/data.py
