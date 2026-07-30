import argparse
import random
from pathlib import Path
 
import cv2
import numpy as np
from PIL import Image
 
from common.config import PROCESSED, CLASSES, RANDOM_SEED
 
# Augmentation
# Operates on the already-preprocessed (CLAHE + grayscale + resized) JPEGs,
# same as split.py does. Only label-preserving, MRI-safe transforms:
# small rotation, slight zoom, brightness/contrast jitter, mild noise.
# No horizontal flip, left/right is not guaranteed to be label-preserving
# for brain MRI depending on how the scans were acquired/labeled.
 
def augment_image(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
 
    # Small-angle rotation (+/- 15 deg)
    angle = rng.uniform(-15, 15)
    rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, rot_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
 
    # Slight zoom (+/- 10%)
    scale = rng.uniform(0.9, 1.1)
    zoom_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    img = cv2.warpAffine(img, zoom_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
 
    # Brightness / contrast jitter
    alpha = rng.uniform(0.9, 1.1)  # contrast
    beta = rng.uniform(-10, 10)    # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
 
    # Mild Gaussian noise (applied 50% of the time so not every augmented copy looks equally noisy)
    if rng.random() < 0.5:
        noise = rng_normal_like(img, rng)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
 
    return img
 
 
def rng_normal_like(img: np.ndarray, rng: random.Random) -> np.ndarray:
    seed = rng.randint(0, 2**31 - 1)
    return np.random.default_rng(seed).normal(0, 3, img.shape).astype(np.float32)
 
 
def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))
 
 
def save_gray(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(path, format="JPEG", quality=95)
 
 
#  Per-client, per-class top-up 
# For each client: copy its data as-is, then for any class below that
# CLIENT's own max-class count, generate augmented variants of that
# client's own images until it catches up. A class with zero real samples
# for a client is left at zero — that's a genuine missing-class scenario,
# not something augmentation can or should manufacture. This keeps the
# federated "no data leaves the client" assumption intact and preserves
# the non-IID *shape* (who's glioma-heavy vs meningioma-heavy) instead of
# quietly flattening it into IID.
 
def augment_client_folder(src_root: Path, dst_root: Path, seed: int) -> None:
    rng = random.Random(seed)
    client_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
 
    for client_dir in client_dirs:
        files_by_class: dict[str, list[Path]] = {}
        counts_before: dict[str, int] = {}
 
        for cls in CLASSES:
            cls_dir = client_dir / cls
            files = sorted(cls_dir.glob("*.jpg")) if cls_dir.exists() else []
            files_by_class[cls] = files
            counts_before[cls] = len(files)
 
        nonzero_counts = [c for c in counts_before.values() if c > 0]
        target = max(nonzero_counts) if nonzero_counts else 0
 
        for cls in CLASSES:
            files = files_by_class[cls]
            out_dir = dst_root / client_dir.name / cls
 
            #    in split.py but keeps this script self-contained)
            for f in files:
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.open(f).convert("L").save(out_dir / f.name, format="JPEG", quality=95)
 
            if not files:
                continue  # true vacant class for this client — nothing to augment from
 
            deficit = target - len(files)
            for i in range(deficit):
                src_file = files[i % len(files)]
                img = load_gray(src_file)
                aug = augment_image(img, rng)
                out_name = f"{src_file.stem}_aug{i}.jpg"
                save_gray(aug, out_dir / out_name)
 
        counts_after = {
            cls: len(list((dst_root / client_dir.name / cls).glob("*.jpg")))
            for cls in CLASSES
        }
        print(f"  {client_dir.name}: before={counts_before}  after={counts_after}")
 
 
#entry point 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment existing client splits to reduce per-client class imbalance "
                     "(run split.py first — this reads clients_N_iid / clients_N_noniid "
                     "and writes clients_N_augmented_iid / clients_N_augmented_noniid)"
    )
    parser.add_argument(
        "--clients",
        type=int,
        required=True,
        help="Number of clients — must match an existing clients_N_* split from split.py",
    )
    parser.add_argument(
        "--mode",
        choices=["iid", "noniid", "both"],
        default="both",
        help="Which existing split(s) to augment",
    )
    return parser.parse_args()
 
 
def main() -> None:
    args = parse_args()
    n = args.clients
    modes = ["iid", "noniid"] if args.mode == "both" else [args.mode]
 
    for mode in modes:
        src_root = PROCESSED / f"clients_{n}_{mode}"
        dst_root = PROCESSED / f"clients_{n}_augmented_{mode}"
 
        if not src_root.exists():
            print(f"[ERROR] {src_root} not found — run split.py --clients {n} --mode {mode} first")
            continue
 
        if dst_root.exists():
            print(f"\n[SKIP] {dst_root.name} already exists — delete it to regenerate")
            continue
 
        print(f"\nAugmenting {src_root.name} → {dst_root.name}")
        augment_client_folder(src_root, dst_root, seed=RANDOM_SEED + 3)
        print(f"\n  Saved → {dst_root}")
 
    print("\nDone.")
    for mode in modes:
        print(f"  clients_{n}_augmented_{mode}/  ← {n}-client augmented-{mode} experiment")
 
 
if __name__ == "__main__":
    main()
 
