
import argparse
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
 
from common.config import (
    RAW_DATA, PROCESSED,
    CLASSES,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
    RANDOM_SEED,
)
 
COL_W = 13
 
def count_folder(folder: Path) -> dict[str, int]:
    """Count .jpg images per class inside a split or client folder."""
    return {
        cls: len(list((folder / cls).glob("*.jpg"))) if (folder / cls).exists() else 0
        for cls in CLASSES
    }
 
 
def print_table(rows: list[tuple[str, dict[str, int]]], title: str) -> None:
    """Print a formatted count table with imbalance ratio."""
    print(f"\n{title}")
    print("─" * (COL_W + COL_W * len(CLASSES) + 8))
 
    header = f"{'':>{COL_W}}" + "".join(f"{c:>{COL_W}}" for c in CLASSES) + f"{'TOTAL':>{COL_W}}"
    print(header)
    print("─" * len(header))
 
    for label, counts in rows:
        total = sum(counts.values())
        row = f"{label:>{COL_W}}" + "".join(f"{counts[c]:>{COL_W}}" for c in CLASSES) + f"{total:>{COL_W}}"
        print(row)
 
 
def imbalance_ratio(counts: dict[str, int]) -> float:
    vals = [v for v in counts.values() if v > 0]
    return max(vals) / min(vals) if len(vals) > 1 else 1.0
 
 
 
def check_main_splits() -> None:
    rows = []
    for split in ["Train", "Val", "Test"]:
        counts = count_folder(PROCESSED / split)
        rows.append((split, counts))
 
    print_table(rows, "PREPROCESSED DATASET — TRAIN / VAL / TEST")
 
    train_counts = count_folder(PROCESSED / "Train")
    ratio = imbalance_ratio(train_counts)
    symbol = "[WARN]" if ratio > 1.3 else "[ OK ]"
    print(f"\n  {symbol} Train class imbalance ratio: {ratio:.2f}x", end="")
    if ratio > 1.3:
        print(" — consider weighted loss or oversampling")
    else:
        print(" — acceptable")
 
 
 
def check_client_experiments() -> None:
    
    # Find all client experiment folders
    experiment_dirs = sorted([
        d for d in PROCESSED.iterdir()
        if d.is_dir() and d.name.startswith("clients_")
    ])
 
    if not experiment_dirs:
        print("\n  [INFO] No client experiments found — run split.py first")
        return
 
    for exp_dir in experiment_dirs:
        client_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
 
        if not client_dirs:
            continue
 
        rows = []
        all_counts = []
 
        for client_dir in client_dirs:
            counts = count_folder(client_dir)
            rows.append((client_dir.name, counts))
            all_counts.append(counts)
 
        print_table(rows, f"CLIENT EXPERIMENT — {exp_dir.name}")
 
        # Per-client imbalance ratios
        print()
        for (label, counts), client_dir in zip(rows, client_dirs):
            ratio = imbalance_ratio(counts)
            symbol = "[WARN]" if ratio > 2.0 else "[ OK ]"
            print(f"  {symbol} {label} imbalance ratio: {ratio:.2f}x")
 
        # Cross-client balance — how evenly is each class distributed across clients?
        print()
        for cls in CLASSES:
            class_counts = [c[cls] for c in all_counts]
            if sum(class_counts) == 0:
                continue
            mx, mn = max(class_counts), min(class_counts)
            cross_ratio = mx / mn if mn > 0 else float("inf")
            symbol = "[WARN]" if cross_ratio > 3.0 else "[ OK ]"
            print(f"  {symbol} {cls} cross-client ratio: {cross_ratio:.2f}x  {class_counts}")
 
 
 
def make_comparison_grid() -> None:

    rng = random.Random(RANDOM_SEED)
    samples = []
 
    for cls in CLASSES:
        raw_dir = RAW_DATA / "Train" / cls
        pre_dir = PROCESSED / "Train" / cls
        if not raw_dir.exists() or not pre_dir.exists():
            continue
 
        raw_files = [f for f in raw_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not raw_files:
            continue
 
        chosen_raw = rng.choice(raw_files)
        chosen_pre = pre_dir / (chosen_raw.stem + ".jpg")
        if not chosen_pre.exists():
            continue
 
        samples.append((cls, chosen_raw, chosen_pre))
 
    if not samples:
        print("  [WARN] No raw+preprocessed pairs found — run preprocess.py first")
        return
 
    cell   = 224
    pad    = 8
    lbl_h  = 20
    n_rows = len(samples)
    n_cols = 2
 
    grid_w = n_cols * cell + (n_cols + 1) * pad
    grid_h = n_rows * (cell + lbl_h) + (n_rows + 1) * pad
    grid   = np.full((grid_h, grid_w), 200, dtype=np.uint8)
 
    for row, (cls, raw_path, pre_path) in enumerate(samples):
        y0 = pad + row * (cell + lbl_h + pad)
 
        # Original
        img_raw = np.array(Image.open(raw_path).convert("L").resize((cell, cell), Image.LANCZOS))
        x0_raw  = pad
        grid[y0: y0 + cell, x0_raw: x0_raw + cell] = img_raw
 
        # Preprocessed
        img_pre = np.array(Image.open(pre_path).convert("L"))
        x0_pre  = pad + cell + pad
        grid[y0: y0 + cell, x0_pre: x0_pre + cell] = img_pre
 
        # Labels
        label_y = y0 + cell + 2
        cv2.putText(grid, f"{cls} — original",   (x0_raw, label_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, 0, 1, cv2.LINE_AA)
        cv2.putText(grid, "after CLAHE",          (x0_pre, label_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, 0, 1, cv2.LINE_AA)
 
    out_path = PROCESSED / "sample_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"  Comparison grid saved → {out_path}")
 
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Verify preprocessed dataset")
    parser.add_argument("--visual", action="store_true",
                        help="Generate before/after CLAHE comparison grid")
    args = parser.parse_args()
 
    print("═" * (COL_W + COL_W * len(CLASSES) + 8))
    check_main_splits()
    check_client_experiments()
    print()
 
    if args.visual:
        print("Generating visual comparison grid …")
        make_comparison_grid()
 
 
if __name__ == "__main__":
    main()
