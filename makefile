all: extract train demo

extract:
	python3 src/data_extraction.py

train:
	python3 src/model_training.py

demo:
	python3 src/fingers.py

compress:
	./compress_videos.sh