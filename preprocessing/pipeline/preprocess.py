# Read image from the folder and apply  CLAHE+ minmax normalization

import cv2
import csv
import numpy as np
from pathlib import Path
from PIL import Image

from config import(
    RAW_DATA,PROCESSED,
    CLASSES, SPLITS, VALID_EXTS,
    IMG_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID
)

def apply_clahe (img_gray:np.ndarray)->np.ndarray:
    #Applying CLAHE to a single channel image

    clahe = cv2.createCLAHE(
        clipLimit = CLAHE_CLIP_LIMIT,
        tileGridSize = CLAHE_TILE_GRID,
    )
    return clahe.apply(img_gray)

def normalize_to_float(img_gray:np.ndarray)->np.ndarray:
    #Normalizing to [0,1]
    #Dividing my 255 since image is uint8
    return img_gray.astype(np.float32)/255.0

def float_to_uint8(img_float: np.ndarray)-> np.ndarray:
    #Float back to image
    return (img_float*255).clip(0,255).astype(np.uint8)

def process_image(src_path:Path,dst_path:Path)->bool:
    #Full pipeline for an image

    img = Image.open(src_path)

    if img.mode!= "L":
        img = img.convert("L")

    img = img.resize(IMG_SIZE, Image.LANCZOS)

    img_np = np.array(img,dtype = np.uint8)

    img_clahe = apply_clahe(img_np)

    img_float = normalize_to_float(img_clahe)

    img_uint8 = float_to_uint8(img_float)

    dst_path.parent.mkdir(parents = True, exist_ok = True)

    dst_path = dst_path.with_suffix(".jpg")

    Image.fromarray(img_uint8, mode = "L").save(dst_path,format = "JPEG", quality = 95)
    return True

def run():
    total_ok = 0
    total_skipped = 0

    for split in SPLITS:
        dst_dir = PROCESSED / split
        dst_dir.mkdir(parents=True, exist_ok=True)


        for cls in CLASSES:
            src_dir = RAW_DATA / split / cls

            if not src_dir.exists():
                print(f"Source folder missing {src_dir}")
                continue

            files = [f for f in src_dir.iterdir() if f.suffix in VALID_EXTS]

            ok = 0
            skipped = 0

            for f in files:
                # Preserve the class directory
                dst = dst_dir / cls / f.stem

                try:
                    process_image(f, dst)
                    ok += 1

                except Exception as e:
                    print(f"Skipped {f.name}: {e}")
                    skipped += 1

            print(f"{split}/{cls}: {ok} processed, {skipped} skipped")

            total_ok += ok
            total_skipped += skipped
            
    print(f"\nDone. {total_ok} images saved to {PROCESSED}")

    if total_skipped:
        print(f"{total_skipped} images skipped")
if __name__ =="__main__":
    run()