#!/usr/bin/env python3
"""Train the chicken-label detector with a separately downloaded dataset."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "data.yaml"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 60-class chicken-label detector."
    )
    parser.add_argument(
        "--model",
        default="yolo11m.pt",
        help="Ultralytics checkpoint name or local .pt path (default: %(default)s).",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Dataset YAML template containing splits and class names.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root of the separately downloaded YOLO dataset.",
    )
    parser.add_argument("--device", default="auto", help="For example: auto, cpu, 0, or 0,1.")
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=150)
    return parser.parse_args()


def make_runtime_config(template: Path, data_root: Path) -> Path:
    """Create a temporary YAML with an absolute dataset root for this machine."""
    if not template.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {template}")
    if not data_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist: {data_root}\n"
            "Download the public dataset or pass --data-root."
        )

    lines = template.read_text(encoding="utf-8").splitlines()
    replacement = f"path: {json.dumps(data_root.resolve().as_posix(), ensure_ascii=False)}"
    for index, line in enumerate(lines):
        if line.lstrip().startswith("path:"):
            lines[index] = replacement
            break
    else:
        lines.insert(0, replacement)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", encoding="utf-8", delete=False
    ) as stream:
        stream.write("\n".join(lines) + "\n")
        return Path(stream.name)


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.epochs <= 0 or args.patience < 0:
        raise SystemExit("--batch and --epochs must be positive; --patience cannot be negative")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    runtime_config = make_runtime_config(
        args.data_config.resolve(), args.data_root.resolve()
    )
    try:
        train_args = {
            "data": str(runtime_config),
            "task": "detect",
            "mode": "train",
            "imgsz": 1120,
            "max_det": 300,
            "batch": args.batch,
            "epochs": args.epochs,
            "patience": args.patience,
            "scale": 0.5,
            "degrees": 180,
            "fliplr": 0.25,
            "flipud": 0.25,
            "seed": args.seed,
            "deterministic": True,
        }
        if args.device != "auto":
            train_args["device"] = args.device
        YOLO(args.model).train(**train_args)
    finally:
        runtime_config.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
