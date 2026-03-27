#!/usr/bin/env python3
"""Generate quality benchmark charts for TurboQuant."""

import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")

# --- Data from benchmark runs ---

# MSE Distortion (from "quality MSE distortion" test)
dims_mse = [64, 128, 256, 512, 1024]
raw_mse = [0.000832, 0.000447, 0.000242, 0.000125, 0.000065]
polar_only_mse = [0.068317, 0.068261, 0.070071, 0.069677, 0.070308]
paper_bound = [0.021094] * 5  # D_mse <= 2.7 * (1/4^3.5)

# Inner Product Distortion (from "quality inner product distortion" test)
dims_dot = [64, 128, 256, 512, 1024]
full_sq_err = [0.000754, 0.000469, 0.000243, 0.000145, 0.000051]
polar_sq_err = [0.000935, 0.000553, 0.000278, 0.000159, 0.000054]

# Component Analysis (from "quality component analysis" test)
dims_comp = [128, 256, 512, 1024]
comp_polar_mse = [0.000533, 0.000275, 0.000135, 0.000068]
comp_full_mse = [0.000446, 0.000242, 0.000124, 0.000064]
improvement_pct = [16.3, 11.7, 8.4, 6.0]

# Recall@k (from quality.zig runs)
dims_recall = [64, 128, 256, 512]
recall_full = [0.98, 1.00, 0.98, 0.98]
recall_polar = [0.94, 1.00, 0.98, 0.98]
dot_error = [5.24e-3, 4.26e-3, 3.16e-3, 2.58e-3]

# --- Chart style matching existing charts ---
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
}
plt.rcParams.update(STYLE)


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


# --- Chart 1: MSE Distortion ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dims_mse, raw_mse, "o-", color="#4285F4", linewidth=2, label="Full (polar+QJL)", markersize=6)
ax.plot(dims_mse, polar_only_mse, "s--", color="#EA4335", linewidth=2, label="Polar only (normalized)", markersize=6)
ax.plot(dims_mse, paper_bound, "d:", color="#34A853", linewidth=2, label="Paper bound (b=3.5)", markersize=6)
for i, d in enumerate(dims_mse):
    ax.annotate(f"{raw_mse[i]:.1e}", (d, raw_mse[i]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8, color="#4285F4")
ax.set_xlabel("Dimension")
ax.set_ylabel("MSE")
ax.set_title("MSE Distortion by Dimension")
ax.set_yscale("log")
ax.legend(loc="upper right")
save(fig, "mse-distortion.png")


# --- Chart 2: Dot Product Error ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dims_dot, full_sq_err, "o-", color="#4285F4", linewidth=2, label="Full (polar+QJL)", markersize=6)
ax.plot(dims_dot, polar_sq_err, "s--", color="#EA4335", linewidth=2, label="Polar only", markersize=6)
for i, d in enumerate(dims_dot):
    ax.annotate(f"{full_sq_err[i]:.1e}", (d, full_sq_err[i]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8, color="#4285F4")
ax.set_xlabel("Dimension")
ax.set_ylabel("Mean Squared Error")
ax.set_title("Inner Product Distortion by Dimension")
ax.set_yscale("log")
ax.legend()
save(fig, "dot-product-error.png")


# --- Chart 3: Component Analysis (grouped bar) ---
fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(dims_comp))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], [m * 1000 for m in comp_polar_mse],
               width, label="Polar only", color="#FBBC04", edgecolor="white")
bars2 = ax.bar([i + width/2 for i in x], [m * 1000 for m in comp_full_mse],
               width, label="Polar + QJL", color="#4285F4", edgecolor="white")
# Add improvement labels
for i, pct in enumerate(improvement_pct):
    ax.annotate(f"-{pct:.0f}%", (i + width/2, comp_full_mse[i] * 1000),
                textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9,
                fontweight="bold", color="#34A853")
ax.set_xlabel("Dimension")
ax.set_ylabel("MSE (x10^-3)")
ax.set_title("Component Analysis: QJL Improvement")
ax.set_xticks(x)
ax.set_xticklabels(dims_comp)
ax.legend()
save(fig, "component-analysis.png")


# --- Chart 4: Recall@k ---
fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(dims_recall))
width = 0.35
ax.bar([i - width/2 for i in x], recall_full, width,
       label="Polar + QJL", color="#4285F4", edgecolor="white")
ax.bar([i + width/2 for i in x], recall_polar, width,
       label="Polar only", color="#FBBC04", edgecolor="white")
for i, r in enumerate(recall_full):
    ax.annotate(f"{r:.2f}", (i - width/2, r), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=10, fontweight="bold")
for i, r in enumerate(recall_polar):
    ax.annotate(f"{r:.2f}", (i + width/2, r), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Dimension")
ax.set_ylabel("Recall@10")
ax.set_title("Recall@10 (N=1000, 50 queries)")
ax.set_xticks(x)
ax.set_xticklabels(dims_recall)
ax.set_ylim(0, 1.15)
ax.legend()
save(fig, "recall-at-k.png")


print("\nAll charts generated.")
