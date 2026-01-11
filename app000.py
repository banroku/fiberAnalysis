
# app.py
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from fiberlen.pipeline import RunParams, run_pipeline


st.set_page_config(page_title="Fiber Length Analyzer", layout="wide")

st.title("Fiber Length Analyzer (minimal)")

uploaded = st.file_uploader("Select skeleton tif/tiff", type=["tif", "tiff"])
if uploaded is None:
    st.stop()

# --- minimal params (later expandable) ---
with st.sidebar:
    st.header("Parameters (minimal)")
    border = st.number_input("border_margin_px", min_value=0, max_value=50, value=5, step=1)
    exclude_border = st.checkbox("exclude_border_touching", value=True)
    bins = st.slider("hist_bins", min_value=5, max_value=100, value=30, step=1)
    round_px = st.slider("discrete_round_px", min_value=1, max_value=50, value=5, step=1)

    probe_len = st.slider("pairing_probe_len", min_value=3, max_value=50, value=12, step=1)
    max_cost = st.slider("pairing_max_cost_accept", min_value=0.05, max_value=1.0, value=0.35, step=0.01)

params = RunParams(
    border_margin_px=int(border),
    exclude_border_touching=bool(exclude_border),
    hist_bins=int(bins),
    discrete_round_px=int(round_px),
    probe_len=int(probe_len),
    max_cost_accept=float(max_cost),
)

# Save upload to temp file (imageio expects a path)
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / uploaded.name
    p.write_bytes(uploaded.getvalue())

    out = run_pipeline(p, params)
    mask = out["mask"]
    lengths = out["lengths_px"]
    meta = out["meta"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Overlay (top fibers)")
    # Simple overlay: draw all fibers in thin lines; later you can draw top-N etc.
    fig = plt.figure()
    plt.imshow(mask.astype(np.uint8), cmap="gray", interpolation="nearest")
    # draw a subset for speed
    # (minimal: draw nothing fancy yet)
    plt.title(f"fibers_final={meta['fibers_final']}")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Histogram (count + length-weighted)")
    if len(lengths) == 0:
        st.warning("No lengths found after filtering.")
    else:
        counts, edges = np.histogram(lengths, bins=params.hist_bins)
        weights, _ = np.histogram(lengths, bins=params.hist_bins, weights=lengths)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig2 = plt.figure()
        ax1 = plt.gca()
        ax1.bar(centers, counts, width=float(edges[1] - edges[0]))
        ax1.set_xlabel("fiber length (px)")
        ax1.set_ylabel("count")
        ax2 = ax1.twinx()
        ax2.plot(centers, weights, marker="o", linewidth=1.5)
        ax2.set_ylabel("sum length (px)")
        st.pyplot(fig2)
        plt.close(fig2)

st.subheader("Discrete (count + length-weighted)")
if len(lengths) > 0:
    step = params.discrete_round_px
    rounded = (np.round(lengths / step) * step).astype(int)
    xs = np.unique(rounded)
    ys_count = np.array([(rounded == x).sum() for x in xs], dtype=int)
    ys_sum = np.array([lengths[rounded == x].sum() for x in xs], dtype=float)

    fig3 = plt.figure()
    ax1 = plt.gca()
    ax1.plot(xs, ys_count, marker="o", linewidth=1.5)
    ax1.set_xlabel(f"fiber length (px) rounded to {step}px")
    ax1.set_ylabel("count")
    ax2 = ax1.twinx()
    ax2.plot(xs, ys_sum, marker="o", linewidth=1.5)
    ax2.set_ylabel("sum length (px)")
    st.pyplot(fig3)
    plt.close(fig3)

st.caption("This is a minimal app. Later we can add full overlay of top-N fibers and export buttons.")
