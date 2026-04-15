from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen


BASE_URL = "https://raw.githubusercontent.com/tangsanli5201/DeepPCB/master/PCBData/"


def download(url: str, target: Path, retries: int) -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                target.write_bytes(response.read())
            return
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Failed to download {url}") from last_error


def fetch_split_file(root: Path, name: str) -> Path:
    target = root / name
    download(f"{BASE_URL}{name}", target, retries=3)
    return target


def build_tasks(root: Path) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for name in ("trainval.txt", "test.txt"):
        split_file = fetch_split_file(root, name)
        for raw_line in split_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            image_rel, label_rel = line.split()
            image_rel = image_rel.replace(".jpg", "_test.jpg")
            tasks.append((f"{BASE_URL}{image_rel}", root / image_rel))
            tasks.append((f"{BASE_URL}{label_rel}", root / label_rel))
    unique_tasks: dict[str, Path] = {}
    for url, target in tasks:
        unique_tasks[url] = target
    return list(unique_tasks.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DeepPCB images and labels from the official GitHub mirror.")
    parser.add_argument("--out", type=Path, default=Path("datasets/deeppcb/PCBData"), help="Output PCBData directory.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download workers.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per file.")
    args = parser.parse_args()

    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(root)
    ok = 0
    skipped = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, url, target, args.retries): (url, target) for url, target in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            _, target = futures[future]
            try:
                before = target.exists() and target.stat().st_size > 0
                future.result()
                if before:
                    skipped += 1
                else:
                    ok += 1
            except Exception as exc:  # pragma: no cover - network retry path
                failed += 1
                print(f"[FAIL] {target}: {exc}", file=sys.stderr)
            if index % 200 == 0:
                print(f"progress {index}/{len(tasks)} ok={ok} skipped={skipped} failed={failed}", flush=True)
    print(f"done ok={ok} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
