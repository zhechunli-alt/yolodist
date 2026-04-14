from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.train.ultralytics_runner import run_predict


def main() -> None:
    run_predict(
        model_path=ROOT / "weights" / "yolo11n.pt",
        image_path=ROOT / "bus.jpg",
        project=ROOT / "runs" / "baseline",
        name="predict",
    )


if __name__ == "__main__":
    main()
