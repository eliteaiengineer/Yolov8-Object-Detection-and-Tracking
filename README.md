# 🔭 YOLOv8 Finetuning Template

Minimal, production-style template to **finetune YOLOv8** on your custom dataset using the official [`ultralytics`](https://github.com/ultralytics/ultralytics) package.

- ✅ Finetune from a pretrained checkpoint (`yolov8n.pt` by default)
- ✅ Handles runs/weights under `./outputs/`
- ✅ Simple inference script for images/folders/videos
- ✅ Works from repo root with src-layout
- ✅ Pytest smoke test (imports)

---

## 🚀 Quickstart

1) **Install**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

| If you have a CUDA GPU, install the matching PyTorch wheel first from pytorch.org, then pip install ultralytics.

2) Prepare dataset (YOLO format) under data/datasets/custom/:

data/datasets/custom/
├─ data.yaml
├─ images/train/*.jpg  (or .png)
├─ images/val/*.jpg
├─ labels/train/*.txt  # same stem as images
└─ labels/val/*.txt

Example data.yaml:

```# data/datasets/custom/data.yaml
path: data/datasets/custom
train: images/train
val: images/val
names:
  0: class0
  1: class1
```
3) Train:


```bash
python3 -m src.yolov8_ft.train \
  --data data/datasets/custom/data.yaml \
  --weights yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device cpu          # or 'cuda', 'mps'
```

4) Predict

```bash
python3 -m src.yolov8_ft.predict \
  --weights outputs/train/last.pt \
  --source data/datasets/custom/images/val \
  --save-txt
```

5) Export (optional)

```bash
yolo export model=outputs/train/best.pt format=onnx imgsz=640
```

### Makefile shortcuts

```bash 
make train       # quick demo run
make predict     # runs inference on val images
make test        # smoke tests (imports)
```

### Tests
| Make sure to prepend PYTHONPATH so src-layout imports resolve:

```bash
PYTHONPATH=src pytest -v
```

### 🧠 Notes

- Default model: yolov8n.pt (fastest). Swap to yolov8s.pt, yolov8m.pt, etc. as needed.
- All runs/models/predictions are written under ./outputs/ to keep the repo self-contained.
- You can resume training by passing --resume outputs/train/last.pt.

