all: extract train demo

extract:
	python3 src/data_extraction.py

train:
	python3 src/model_training.py

demo:
	python3 src/fingers.py

report:
	mkdir -p project
	cp src report.pdf readme.md project