#Shared configuration for the pipeline


from pathlib import Path

#Paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT/ "dataset" / "Epic and CSCR hospital Dataset"
PROCESSED = ROOT / "dataset" / "preprocessed"

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
SPLITS = ["Train","Test"]
VALID_EXTS = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}


IMG_SIZE = (244,244)

CLAHE_CLIP_LIMIT = 2.0 
CLAHE_TITLE_GRID = (8,8)

VAL_FRACTION = 0.1
RANDOM_SEED = 42
#For 3,4 and 5 clients
CLIENTS_COUNT = [3,4,5]

SPLIT_MODES = ["iid", "noniid"]

NONIID_ALPHA = 0.5