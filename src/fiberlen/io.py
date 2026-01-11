# src/fiberlen/io.py

from pathlib import Path
from dataclasses import dataclass
from typing import Union

import numpy as np
import imageio.v3 as iio


@dataclass(frozen=True)
class SkeletonImage:
    mask: np.ndarray   # bool array, True = skeleton pixel
    path: Path


def read_skeleton_tif(
    path: Union[str, Path],
    *,
    foreground: str = "auto",
    pad: int = 0,
) -> SkeletonImage:
    """
    Read skeleton TIFF and return boolean mask.
    """
    p = Path(path)

    arr = iio.imread(p)

    # --- ensure 2D ---
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = (
                0.299 * arr[:, :, 0]
                + 0.587 * arr[:, :, 1]
                + 0.114 * arr[:, :, 2]
            ).astype(arr.dtype)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got {arr.shape}")

    # --- convert to uint8 if needed ---
    if arr.dtype == np.bool_:
        mask = arr.copy()
    else:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(np.nanmax(arr))
            if mx <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        nonzero = np.count_nonzero(arr)
        total = arr.size

        if foreground == "white":
            mask = arr > 0
        elif foreground == "black":
            mask = arr == 0
        elif foreground == "auto":
            # skeleton is usually sparse
            if nonzero / total < 0.5:
                mask = arr > 0
            else:
                mask = arr == 0
        else:
            raise ValueError("foreground must be 'auto', 'white', or 'black'")

    if pad > 0:
        mask = np.pad(mask, pad_width=pad, mode="constant", constant_values=False)

    return SkeletonImage(mask=mask.astype(bool), path=p)
