install:
	pip install -r requirements.txt
train:
	python3 -m src.yolov8_ft.train \
		--data data/datasets/custom/data.yaml \
		--weights yolov8n.pt \
		--epochs 1 \
		--imgsz 640 \
		--batch 8 \
		--device cpu

predict:
	python3 -m src.yolov8_ft.predict \
		--weights outputs/train/last.pt \
		--source data/datasets/custom/images/val \
		--save-txt

test:
	PYTHONPATH=src pytest -v
