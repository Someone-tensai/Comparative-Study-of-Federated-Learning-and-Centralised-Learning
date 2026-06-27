import os
from pathlib import Path
from PIL import Image

DATA_ROOT = Path("dataset/Epic and CSCR hospital Dataset")
CLASSES = ["glioma","meningioma","pituitary","notumor"]

for split in ["Train","Test"]:
    for cls in CLASSES:
        folder = DATA_ROOT / split / cls

        #Path of every files in the folder
        files = list(folder.iterdir()) if folder.exists() else []
        
        valid, corrupted = 0,[]
        #Allowing only appropriate format
        for f in files:
            if f.suffix.lower() not in [".jpg",".png",".jpeg"]:
                continue
            try:
                img = Image.open(f)
                img.verify()
                valid+=1
            except Exception:
                corrupted.append(f.name)
        print(f"{split}/{cls} : {valid} valid, {len(corrupted)} corrupt")
        if corrupted:
            print(f"Corrupt: {corrupted}")