from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO
from .utils import ensure_dir, repo_outputs


def main() -> int:
    ap = argparse.ArgumentParser(description="Run YOLOv8 inference.")
    ap.add_argument("--weights", type=str, required=True, help="Path to model .pt (e.g., outputs/train/best.pt)")
    ap.add_argument("--source", type=str, required=True, help="Image/video/dir path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--save-txt", action="store_true", help="Save YOLO txt predictions")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--name", type=str, default="predict")
    args = ap.parse_args()

    model = YOLO(args.weights)

    proj = repo_outputs()
    out_dir = ensure_dir(proj / args.name)

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,           # save images with boxes
        save_txt=args.save_txt,
        project=str(proj),
        name=args.name,
        exist_ok=True,
    )
    print(results)
    print(f"✅ Predictions saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
