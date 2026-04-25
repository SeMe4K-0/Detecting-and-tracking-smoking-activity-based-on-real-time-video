"""
Export trained YOLO26n model to CoreML (.mlpackage) for iOS deployment.

Usage:
  python export_coreml.py --weights runs/detect/cigarette_yolo26n/weights/best.pt

The output cigarette_yolo26n.mlpackage should be placed in:
  SmokingDetectorApp/SmokingDetectorApp/Resources/
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


def export(weights_path: str):
    model = YOLO(weights_path)

    # NMS=True embeds Non-Maximum Suppression into the model graph.
    # iOS receives VNRecognizedObjectObservation directly — no manual NMS needed.
    model.export(
        format="coreml",
        nms=True,
        imgsz=640,
    )

    # Ultralytics saves as best.mlpackage next to weights
    src = Path(weights_path).with_suffix(".mlpackage")
    if not src.exists():
        # Try sibling directory
        src = Path(weights_path).parent / "best.mlpackage"

    dst = Path("cigarette_yolo26n.mlpackage")
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Exported: {dst.resolve()}")
        print("Copy it to SmokingDetectorApp/SmokingDetectorApp/Resources/")
    else:
        print(f"Export complete. Locate .mlpackage near {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="runs/detect/runs/cigarette_yolo26n/weights/best.pt",
        help="Path to trained .pt weights file",
    )
    args = parser.parse_args()
    export(args.weights)
