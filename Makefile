.PHONY: install
install:
	pip install -e .

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: pdfs-to-images
pdfs-to-images:
	python ./score_retrieval/migration.py

.PHONY: copy-data
copy-data:
	python ./score_retrieval/copying.py

.PHONY: delete-images
delete-images:
	find . -name '*.png' -delete
