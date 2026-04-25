import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image, ImageDraw


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_box_xyxy(row: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, row[:4])

    # Heuristic: some exports can return normalized coords in [0,1].
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    x1 = _clip(x1, 0, width - 1)
    x2 = _clip(x2, 0, width - 1)
    y1 = _clip(y1, 0, height - 1)
    y2 = _clip(y2, 0, height - 1)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def run(model_path: str, image_path: str, conf_threshold: float, max_print: int) -> None:
    model = ct.models.MLModel(model_path)
    spec = model.get_spec()
    input_name = spec.description.input[0].name

    original = Image.open(image_path).convert("RGB")
    resized = original.resize((640, 640))

    pred = model.predict({input_name: resized})
    output_name = next(iter(pred))
    output = np.array(pred[output_name])

    if output.ndim != 3 or output.shape[-1] < 6:
        raise RuntimeError(f"Unexpected output shape: {output.shape}. Expected [1, N, 6].")

    rows = output[0]
    conf = rows[:, 4]
    keep = rows[conf >= conf_threshold]

    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Output tensor: {output_name} shape={tuple(output.shape)}")
    print(f"Threshold: {conf_threshold}")
    print(f"Detections >= threshold: {len(keep)}")

    # Print top detections by confidence.
    top = keep[np.argsort(-keep[:, 4])] if len(keep) else keep
    for i, row in enumerate(top[:max_print], start=1):
        x1, y1, x2, y2 = row[:4]
        score = float(row[4])
        cls = int(row[5])
        print(
            f"{i:02d}. cls={cls} conf={score:.3f} "
            f"raw_box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

    # Draw detections on resized image for quick visual verification.
    draw = ImageDraw.Draw(resized)
    for row in top:
        x1, y1, x2, y2 = _to_box_xyxy(row, width=resized.width, height=resized.height)
        score = float(row[4])
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        draw.text((x1 + 3, max(0, y1 - 12)), f"{score:.2f}", fill="lime")

    out_path = Path(image_path).with_name(f"{Path(image_path).stem}_coreml_pred.png")
    resized.save(out_path)
    print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CoreML model and print detections.")
    parser.add_argument("--model", default="cigarette_yolo26n.mlpackage", help="Path to .mlpackage")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max-print", type=int, default=20, help="Max detections to print")
    args = parser.parse_args()

    run(args.model, args.image, args.conf, args.max_print)
