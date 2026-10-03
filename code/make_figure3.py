"""Figure 3 - model comparison (held-out test set and repeated cross-validation).
Reads results/statistics_summary.json written by the pipeline,
so every plotted number comes from the same run as Tables 2 and 3."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
S = json.loads((R / "statistics_summary.json").read_text())
MODELS = ["Random Forest", "XGBoost"]
COLORS = {"Random Forest": "#2a78d6", "XGBoost": "#eb6834"}   # validated categorical pair (CVD dE 24.7)
HATCH = {"Random Forest": "", "XGBoost": "////"}                # secondary encoding for greyscale print
METRICS = [("accuracy", "Accuracy"), ("f1_weighted", "Weighted F1"), ("top_3_accuracy", "Top-3 accuracy")]
TEXT, MUTED, GRID = "#1f1f1f", "#55534e", "#e6e5e1"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": TEXT, "ytick.color": MUTED, "hatch.linewidth": 0.6})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
x = np.arange(len(METRICS))
w = 0.36

# (a) held-out test set: point estimate, 95% Wilson CI where available (accuracy, Top-3)
ax = axes[0]
for k, m in enumerate(MODELS):
    t = S["test_set"][m]
    vals = [t["accuracy"], t["f1_weighted"], t["top_3_accuracy"]]
    lo = [t["accuracy"] - t["accuracy_95ci"][0], np.nan, t["top_3_accuracy"] - t["top_3_accuracy_95ci"][0]]
    hi = [t["accuracy_95ci"][1] - t["accuracy"], np.nan, t["top_3_accuracy_95ci"][1] - t["top_3_accuracy"]]
    pos = x + (k - 0.5) * (w + 0.02)
    ax.bar(pos, vals, w, color=COLORS[m], hatch=HATCH[m], edgecolor="white", linewidth=0.8, label=m, zorder=2)
    err = np.array([lo, hi])
    ok = ~np.isnan(err[0])
    ax.errorbar(pos[ok], np.array(vals)[ok], yerr=err[:, ok], fmt="none", ecolor=TEXT, elinewidth=1, capsize=3, zorder=3)
    for p_, v_, h_ in zip(pos, vals, hi):
        top = v_ + (0 if np.isnan(h_) else h_)
        ax.text(p_, top + 0.015, f"{v_:.3f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
n_test = S["test_set"]["Random Forest"]["n_test"]
ax.set_title(f"(a) Held-out test set (n = {n_test})", fontsize=11, color=TEXT, loc="left")

# (b) repeated stratified CV: mean +/- SD over folds
ax = axes[1]
cv = S["cross_validation"]
cvkey = {"accuracy": "accuracy", "f1_weighted": "f1_weighted", "top_3_accuracy": "top3_accuracy"}
n_folds = S["model_comparison"]["n_folds"]
for k, m in enumerate(MODELS):
    vals = [cv[m][cvkey[a]]["mean"] for a, _ in METRICS]
    sds = [cv[m][cvkey[a]]["sd"] for a, _ in METRICS]
    pos = x + (k - 0.5) * (w + 0.02)
    ax.bar(pos, vals, w, color=COLORS[m], hatch=HATCH[m], edgecolor="white", linewidth=0.8, label=m, zorder=2)
    ax.errorbar(pos, vals, yerr=sds, fmt="none", ecolor=TEXT, elinewidth=1, capsize=3, zorder=3)
    for p_, v_, s_ in zip(pos, vals, sds):
        ax.text(p_, v_ + s_ + 0.015, f"{v_:.3f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
ax.set_title(f"(b) Repeated 4-fold cross-validation ({n_folds} folds)", fontsize=11, color=TEXT, loc="left")

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in METRICS])
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Score")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
fig.text(0.5, -0.02, "Error bars: (a) 95% Wilson confidence intervals (not defined for weighted F1); "
         "(b) standard deviation across folds.", ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
fig.savefig(R / "model_comparison.png", dpi=600, bbox_inches="tight")
print("saved", R / "model_comparison.png")
