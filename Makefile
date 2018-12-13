.PHONY: install
install:
	pip install -e .

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: migrate
migrate:
	python ./score_retrieval/migration.py

.PHONY: copy-data
copy-data:
	python ./score_retrieval/copying.py
