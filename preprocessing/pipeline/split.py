import argparse
import random
import shutil
import numpy as np
from pathlib import Path

from config import(
    PROCESSED,
    CLASSES,
    VAL_FRACTION,
    RANDOM_SEED,
    NONIID_ALPHA,
)

#Helper

def stratified_sample(
        files:list[Path], fraction:float,seed:int
)-> tuple[list[Path],list[Path]]:
    
    #Split files 
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = max(1,round(len(shuffled) * fraction))
    return shuffled[:n], shuffled[n:]

def copy_files(files: list[Path],dst_dir: Path)-> None:
    dst_dir.mkdir(parents = True, exist_ok = True)
    for f in files:
        shutil.copy2(f,dst_dir/f.name)

def make_val_split()-> dict[str,list[Path]]:
    # 10% from each class in Train to val
    val_exists = all((PROCESSED / "Val" / cls).exists() for cls in CLASSES)

    if val_exists:
        print("Skipping because val split already exists")
        # Still need to return train_remainder so client splits can use it
        train_remainder = {}
        for cls in CLASSES:
            all_train = sorted((PROCESSED / "Train" / cls).glob("*.jpg"))
            val_files = sorted((PROCESSED / "Val" /cls).glob("*.jpg"))
            val_stems = {f.stem for f in val_files}
            
            train_remainder[cls] = [f for f in all_train if f.stem not in val_stems]

            return train_remainder

    print("Val Split")
    train_remainder = {}

    for cls in CLASSES:
        src_dir = PROCESSED / "Train" / cls
        if not src_dir.exists():
            print(f"Src {src_dir} not found Run preprocess.py first")
            continue
        
        files = sorted(src_dir.glob("*.jpg"))
        val,remainder = stratified_sample(files, VAL_FRACTION, RANDOM_SEED)

        copy_files(val,PROCESSED / "Val" / cls)
        train_remainder[cls] = remainder 

        print("{cls}: {len(remainder)} train | {len(val)} val")

        return train_remainder
    
    #IID PARTITIONING

def make_iid_split(
            train_remainder:dict[str,list[Path]],
            n_clients: int,
            out_root: Path
    )-> None:
        # Round-robin shuffle

    print("\n IID split ({n_clients} clients)\n")

    rng = random.Random(RANDOM_SEED + 1)

    for cls in CLASSES:
        files = train_remainder.get(cls,[])
        if not files:
            continue

        shuffled = files[:]
        rng.shuffle(shuffled)

        chunks: list[list[Path]] = [[] for _ in range(n_clients)]
    for i, f in enumerate(shuffled):
        chunks[i % n_clients].append(f)

    for idx, chunk in enumerate(chunks, start=1):
        copy_files(chunk, out_root / f"client_{idx}" / cls)

    sizes = [len(c) for c in chunks]
    print(f"  {cls}: {sizes}")

 
# Non IID (Drichlet partitioning)
 
def make_noniid_split(
    train_remainder: dict[str, list[Path]],
    n_clients: int,
    out_root: Path,
    alpha: float = NONIID_ALPHA,
    min_samples: int = 10,
) -> None:
    """
    Alpha controls skew:
        0.1  → extreme skew (each client dominated by 1-2 classes)
        0.5  → moderate skew (realistic hospital scenario)
        100  → nearly IID
    """

    print(f"\n Non-IID split ({n_clients} clients, alpha={alpha}) ")
    rng_np  = np.random.default_rng(RANDOM_SEED + 2)
    rng     = random.Random(RANDOM_SEED + 2)
 
    for cls in CLASSES:
        files = train_remainder.get(cls, [])
        if not files:
            continue
 
        shuffled = files[:]
        rng.shuffle(shuffled)
        total = len(shuffled)
 
        # Dirichlet gives a probability vector of length n_clients that sums to 1
        # Each value = proportion of this class that goes to that client
        proportions = rng_np.dirichlet(alpha=[alpha] * n_clients)
 
        # Converting proportions to actual counts
        counts = (proportions * total).astype(int)
 
        # Enforce minimum floor per client
        for i in range(n_clients):
            if counts[i] < min_samples:
                counts[i] = min_samples
 

        while counts.sum() > total:
            counts[counts.argmax()] -= 1
 
        # Distributing files according to counts
        cursor = 0
        for idx in range(n_clients):
            chunk = shuffled[cursor : cursor + counts[idx]]
            cursor += counts[idx]
            copy_files(chunk, out_root / f"client_{idx + 1}" / cls)
 
        sizes = counts.tolist()
        print(f"  {cls}: {sizes}  (sum={sum(sizes)}, total={total})")
 
 
# Entry point
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split preprocessed data into Val + federated client partitions"
    )
    parser.add_argument(
        "--clients",
        type=int,
        required=True,
        help="Number of federated clients (e.g. 3, 4, 5 — any number works)",
    )
    parser.add_argument(
        "--mode",
        choices=["iid", "noniid", "both"],
        default="both",
        help="iid: equal class distribution | noniid: Dirichlet skew | both: run both",
    )
    return parser.parse_args()
 
def main() -> None:
    args = parse_args()
    n = args.clients
 
    train_remainder = make_val_split()
 
    if not train_remainder:
        print("[ERROR] No training data found. Run preprocess.py first.")
        return
 
    # Requested mode(s)
    modes = ["iid", "noniid"] if args.mode == "both" else [args.mode]
 
    for mode in modes:
        out_root = PROCESSED / f"clients_{n}_{mode}"
 
        if out_root.exists():
            print(f"\n[SKIP] {out_root.name} already exists — delete it to regenerate")
            continue
 
        if mode == "iid":
            make_iid_split(train_remainder, n, out_root)
        else:
            make_noniid_split(train_remainder, n, out_root)
 
        print(f"\n  Saved → {out_root}")
 
    print(f"\nDone.")
    print(f"  Val/              ← shared across all experiments")
    for mode in modes:
        print(f"  clients_{n}_{mode}/  ← {n}-client {mode} experiment")
 
 
if __name__ == "__main__":
    main()