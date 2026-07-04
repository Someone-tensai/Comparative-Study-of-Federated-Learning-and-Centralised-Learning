#Shared configuration for the pipeline


from pathlib import Path

#Paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT/ "dataset" / "Epic and CSCR hospital Dataset"
PROCESSED = ROOT / "dataset" / "preprocessed"

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
SPLITS = ["Train","Test"]
VALID_EXTS = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}


IMG_SIZE = (224,224)

CLAHE_CLIP_LIMIT = 2.0 
CLAHE_TILE_GRID = (8,8)

VAL_FRACTION = 0.1
RANDOM_SEED = 42

NONIID_ALPHA = 0.5