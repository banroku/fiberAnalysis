# src/fiberlen/config.py
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # --- scale ---
    um_per_px: float = 1.5

    # --- graph ---
    connectivity: int = 8

    # --- pairing ---
    pairing_probe_len: int = 12
    pairing_max_cost_accept: float = 0.35

    # --- io ---
    io_foreground: str = "auto"   # "auto" | "white" | "black"
    io_pad: int = 0

    # --- filtering / QC ---
    border_margin_px: int = 5
    exclude_border_touching: bool = True

    # --- overlay top n ---
    overlay_top_n: int = 15
    overlay_linewidth: float = 1.8
    overlay_dpi: int = 220
    overlay_top_fibers_filename: str = "overlay_top_fibers.png"

    # --- visualization ---
    plot_hist_bins: int = 30
    plot_discrete_round_px: int = 1
    plot_dpi: int = 220
    plot_hist_filename: str = "plot_histogram_counts_and_weight.png"
    plot_discrete_filename: str = "plot_discrete_counts_and_weight.png"
    plot_csv_hist_filename: str = "plot_histogram_table.csv"
    plot_csv_discrete_filename: str = "plot_discrete_table.csv"

CFG = Config()

