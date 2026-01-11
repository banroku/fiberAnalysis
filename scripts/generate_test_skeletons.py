import numpy as np
from pathlib import Path
import imageio.v3 as iio


def blank(h=101, w=101):
    return np.zeros((h, w), dtype=bool)


def draw_line(img, r0, c0, r1, c1):
    n = max(abs(r1 - r0), abs(c1 - c0))
    for i in range(n + 1):
        r = int(round(r0 + (r1 - r0) * i / n))
        c = int(round(c0 + (c1 - c0) * i / n))
        img[r, c] = True


def case_cross():
    img = blank()
    draw_line(img, 10, 50, 90, 50)
    draw_line(img, 50, 10, 50, 90)
    return img


def case_T():
    img = blank()
    draw_line(img, 10, 50, 90, 50)
    draw_line(img, 50, 10, 50, 50)
    return img


def case_Y():
    img = blank()
    draw_line(img, 50, 50, 10, 30)
    draw_line(img, 50, 50, 10, 70)
    draw_line(img, 50, 50, 90, 50)
    return img


def case_double_T():
    img = blank()
    draw_line(img, 10, 50, 50, 50)
    draw_line(img, 50, 10, 50, 50)
    draw_line(img, 50, 55, 90, 55)
    draw_line(img, 70, 55, 70, 90)
    return img


def case_curved_vs_straight():
    img = blank()
    draw_line(img, 10, 50, 90, 50)
    draw_line(img, 50, 10, 50, 40)
    draw_line(img, 50, 40, 48, 60)
    draw_line(img, 48, 60, 50, 90)
    return img


def main():
    out = Path("data/raw/test_skeletons")
    out.mkdir(parents=True, exist_ok=True)

    cases = {
        "cross": case_cross(),
        "T": case_T(),
        "Y": case_Y(),
        "double_T": case_double_T(),
        "curved": case_curved_vs_straight(),
    }

    for name, img in cases.items():
        path = out / f"{name}.tif"
        iio.imwrite(path, img.astype(np.uint8) * 255)
        print("saved", path)


if __name__ == "__main__":
    main()
