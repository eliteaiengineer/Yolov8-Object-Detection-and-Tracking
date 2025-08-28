from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO
from .utils import ensure_dir, repo_outputs


def main() -> int:
    ap = argparse.ArgumentParser(description="Finetune YOLOv8 on a custom dataset.")
    ap.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml")
    ap.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained weights or .pt to resume")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")  # 'cuda', 'mps', or 'cpu'
    ap.add_argument("--project", type=str, default="outputs", help="Project dir for runs")
    ap.add_argument("--name", type=str, default="train", help="Run name under project")
    ap.add_argument("--resume", type=str, default=None, help="Resume from last.pt path")
    args = ap.parse_args()

    # ensure outputs directory
    proj = repo_outputs() if args.project == "outputs" else ensure_dir(args.project)
    run_dir = proj / args.name

    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(proj),
        name=args.name,
        exist_ok=True,
        pretrained=not bool(args.resume),
    )
    print(results)  # dict-like metrics
    # model.save(...) not needed; Ultralytics saves best.pt/last.pt in run_dir
    print(f"✅ Finished. Check: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
