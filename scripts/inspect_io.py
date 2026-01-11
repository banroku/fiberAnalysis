# scripts/inspect_io.py
from pathlib import Path
import numpy as np

from fiberlen.io import read_skeleton_tif
from fiberlen.config import CFG

def main():
    img_path = Path("data/raw/GF-skeltonized.tif")
    sk = read_skeleton_tif(img_path, foreground=CFG.io_foreground, pad=CFG.io_pad)

    mask = sk.mask
    print("Loaded:", sk.path)
    print("Shape:", mask.shape)
    print("dtype:", mask.dtype)
    print("True pixels:", int(mask.sum()))
    print("False pixels:", int(mask.size - mask.sum()))
    print("True fraction:", float(mask.mean()))

    # quick sanity: check original unique if you want
    # (read again without bool conversion)
    # print(np.unique(iio.imread(img_path))[:20])

if __name__ == "__main__":
    main()
