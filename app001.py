# app.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import io
import hashlib

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from fiberlen.pipeline import RunParams, run_pipeline


st.set_page_config(page_title="Fiber Length Analyzer", layout="wide")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _make_overlay_png(mask: np.ndarray, fibers, g, top_n: int, linewidth: float = 1.8) -> bytes:
    fig = plt.figure()
    plt.imshow(mask.astype(np.uint8), cmap="gray", interpolation="nearest")
    # matplotlib default color cycle will color successive polylines
    shown = 0
    for f in fibers:
        if shown >= top_n:
            break
        for sid in f.seg_ids:
            px = g.segments[sid].pixels  # list[(r,c)]
            rr = [p[0] for p in px]
            cc = [p[1] for p in px]
            plt.plot(cc, rr, linewidth=linewidth)
        shown += 1

    plt.axis("off")
    return _fig_to_png_bytes(fig)


def _make_original_png(mask: np.ndarray) -> bytes:
    fig = plt.figure()
    plt.imshow(mask.astype(np.uint8), cmap="gray", interpolation="nearest")
    plt.axis("off")
    return _fig_to_png_bytes(fig)


@st.cache_data(show_spinner=False)
def _cached_run(file_bytes: bytes, params_dict: dict) -> dict:
    # RunParams is dataclass; we pass a dict to make it hashable for cache
    params = RunParams(**params_dict)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "input.tif"
        p.write_bytes(file_bytes)
        out = run_pipeline(p, params)
        # Keep only what we need and make it cache-friendly
        return {
            "mask": out["mask"],
            "graph": out["graph"],
            "fibers": out["fibers"],
            "lengths_px": out["lengths_px"],
            "meta": out["meta"],
        }


st.title("Fiber Length Analyzer")

uploaded = st.file_uploader("Select skeleton tif/tiff", type=["tif", "tiff"])
if uploaded is None:
    st.stop()

file_bytes = uploaded.getvalue()
file_key = _sha256_bytes(file_bytes)

with st.sidebar:
    st.header("Parameters")

    top_n = st.slider("top_n", min_value=1, max_value=200, value=15, step=1)

    st.subheader("Filtering")
    border = st.number_input("border_margin_px", min_value=0, max_value=50, value=5, step=1)
    exclude_border = st.checkbox("exclude_border_touching", value=True)

    st.subheader("Pairing (junction linking)")
    probe_len = st.slider("pairing_probe_len", min_value=3, max_value=50, value=12, step=1)
    max_cost = st.slider("pairing_max_cost_accept", min_value=0.05, max_value=1.0, value=0.35, step=0.01)

    st.subheader("Histogram")
    bins = st.slider("hist_bins", min_value=5, max_value=120, value=30, step=1)

    st.subheader("Scale")
    um_per_px = st.number_input("um_per_px", min_value=0.0001, max_value=1000.0, value=1.0, step=0.1)

params = RunParams(
    border_margin_px=int(border),
    exclude_border_touching=bool(exclude_border),
    probe_len=int(probe_len),
    max_cost_accept=float(max_cost),
    hist_bins=int(bins),
    um_per_px=float(um_per_px),
)

# Keep IO/graph defaults (can be exposed later)
params_dict = asdict(params)

with st.spinner("Running pipeline..."):
    out = _cached_run(file_bytes, params_dict)

mask = out["mask"]
g = out["graph"]
fibers = out["fibers"]
lengths_px = out["lengths_px"]
meta = out["meta"]

# Sort fibers by length_um descending (or length_px if needed)
def _fiber_sort_key(f):
    if f.length_um is not None:
        return float(f.length_um)
    if f.length_px is not None:
        return float(f.length_px)
    return -1.0


fibers_sorted = sorted(fibers, key=_fiber_sort_key, reverse=True)

# Prepare two images: original and overlay
orig_png = _make_original_png(mask)
overlay_png = _make_overlay_png(mask, fibers_sorted, g, top_n=int(top_n), linewidth=1.8)

# Toggle state for parallax-like flipping
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "overlay"

colA, colB = st.columns([1, 1])
with colA:
    if st.button("Toggle view (Original / Top-N overlay)"):
        st.session_state.view_mode = "original" if st.session_state.view_mode == "overlay" else "overlay"
with colB:
    st.write(
        f"fibers_total={meta['fibers_total']}  "
        f"fibers_final={meta['fibers_final']}  "
        f"top_n={top_n}  "
        f"exclude_border={exclude_border}  "
        f"border_margin_px={border}"
    )

# Full-width image area
st.subheader("Image view (flip to compare)")
img_bytes = overlay_png if st.session_state.view_mode == "overlay" else orig_png
st.image(img_bytes, use_container_width=True)

# Histogram: show both as percent of totals, in length units (um)
st.subheader("Histogram (percent of total count and percent of total length)")

if lengths_px.size == 0:
    st.warning("No lengths found after filtering. Try lowering border_margin_px or disabling exclude_border_touching.")
    st.stop()

lengths_um = lengths_px * float(um_per_px)

# Histogram bins in um space
counts, edges = np.histogram(lengths_um, bins=int(bins))
weights, _ = np.histogram(lengths_um, bins=int(bins), weights=lengths_um)

total_count = float(counts.sum()) if counts.sum() > 0 else 1.0
total_weight = float(weights.sum()) if weights.sum() > 0 else 1.0

count_pct = (counts / total_count) * 100.0
weight_pct = (weights / total_weight) * 100.0

centers = 0.5 * (edges[:-1] + edges[1:])

fig_h = plt.figure()
ax = plt.gca()

# User requested "point+line" and different colors
ax.plot(centers, count_pct, marker="o", linewidth=1.5, color="C0", label="count (%)")
ax.plot(centers, weight_pct, marker="o", linewidth=1.5, color="C1", label="length-weighted (%)")

ax.set_xlabel("fiber length (um)")
ax.set_ylabel("percent (%)")
ax.set_ylim(0, max(float(count_pct.max()), float(weight_pct.max())) * 1.1 if max(count_pct.max(), weight_pct.max()) > 0 else 1.0)
ax.legend(loc="upper right")
ax.set_title("Histogram (both series are percent of total)")

st.pyplot(fig_h)
plt.close(fig_h)
