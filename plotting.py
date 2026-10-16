"""
plotting.py
===========
All figures, saved at 600 dpi. Model-specific figures use a two-panel layout,
one panel per selected model, and each subplot carries an UPPERCASE panel
letter identifying the model, e.g. "(A) RF", "(B) SVM".

Axis labels and titles follow the manuscript's existing figures (Figs 3-7).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _letter(ax, i, model_name):
    """Place '(A) model' centred under the subplot."""
    ax.text(0.5, -0.20, f"({C.PANEL_LETTERS[i]}) {model_name}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold")


def _save(fig, outpath):
    fig.savefig(outpath, dpi=C.DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 3 / Fig 5 : horizontal bar importance, two panels (one per model)
# --------------------------------------------------------------------------
def two_panel_barh(data_by_model, label_col, xlabel, suptitle, outpath,
                   order_by_first=True):
    models = list(data_by_model.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(7.2 * len(models), 6.2),
                             squeeze=False)
    axes = axes[0]

    order = None
    if order_by_first:
        first = data_by_model[models[0]].sort_values("mean")
        order = list(first[label_col])

    for i, (m, ax) in enumerate(zip(models, axes)):
        df = data_by_model[m].copy()
        if order is not None:
            df = df.set_index(label_col).reindex(order).reset_index()
        y = np.arange(len(df))
        ax.barh(y, df["mean"], xerr=df["std"], color=C.MODEL_COLORS[i],
                error_kw=dict(ecolor="#333333", lw=1), alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(df[label_col])
        ax.set_xlabel(xlabel)
        _letter(ax, i, m)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    _save(fig, outpath)


# --------------------------------------------------------------------------
# Fig 4 / ablation : vertical grouped bars, two panels
# --------------------------------------------------------------------------
def two_panel_vbar(data_by_model, cat_col, val_col, err_col, ylabel,
                   suptitle, outpath, annotate=True):
    models = list(data_by_model.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 5.4),
                             squeeze=False)
    axes = axes[0]
    for i, (m, ax) in enumerate(zip(models, axes)):
        df = data_by_model[m]
        x = np.arange(len(df))
        ax.bar(x, df[val_col], yerr=df[err_col], color=C.MODEL_COLORS[i],
               error_kw=dict(ecolor="#333333", lw=1), alpha=0.9, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(df[cat_col], rotation=0, fontsize=9)
        ax.set_ylabel(ylabel)
        if annotate:
            for xi, (v, e) in enumerate(zip(df[val_col], df[err_col])):
                ax.text(xi, v + e + 0.01 * max(df[val_col].max(), 1e-6),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        _letter(ax, i, m)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    _save(fig, outpath)


# --------------------------------------------------------------------------
# Fig 6 : per-class F1 vs co-occurrence, two panels (one per model)
# --------------------------------------------------------------------------
def two_panel_scatter(scatter_by_model, pearson_by_model, outpath):
    models = list(scatter_by_model.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 5.2),
                             squeeze=False)
    axes = axes[0]
    for i, (m, ax) in enumerate(zip(models, axes)):
        d = scatter_by_model[m]
        x, y = d["co_occurrence"].to_numpy(), d["f1"].to_numpy()
        ax.scatter(x, y, s=70, color=C.MODEL_COLORS[i],
                   edgecolor="#1A202C", zorder=3)
        # OLS line
        if len(x) > 1:
            b, a = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, b * xs + a, color="#C0392B", lw=2,
                    label=f"OLS fit (Pearson r = {pearson_by_model[m]:.2f})")
            ax.legend(loc="upper right", fontsize=9, frameon=False)
        ax.set_xlabel("Co-occurrence (number of other species sharing the same sites)")
        ax.set_ylabel("Per-class F1 (cross-validated)")
        _letter(ax, i, m)
    fig.suptitle("Per-class F1 versus co-occurrence across species",
                 fontsize=13, fontweight="bold")
    _save(fig, outpath)


# --------------------------------------------------------------------------
# Fig 7 : within/between-site variance decomposition (model-agnostic)
# --------------------------------------------------------------------------
def variance_figure(within_frac, var_partition, meta, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Panel A: within-site variability fraction
    wf = within_frac.sort_values()
    labels_a = [C.PRETTY.get(k, k) for k in wf.index]
    ax = axes[0]
    colors = ["#C0392B" if k == "body_length" else "#95A5A6" for k in wf.index]
    ax.barh(np.arange(len(wf)), wf.values, color=colors)
    ax.set_yticks(np.arange(len(wf)))
    ax.set_yticklabels(labels_a)
    ax.set_xlabel("Fraction of multi-specimen sites in which the feature varies")
    ax.text(0.5, -0.12, "(A) Within-site variability of each feature",
            transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")

    # Panel B: within-site share of variance (continuous)
    vp = var_partition.reindex([c for c in C.CONTINUOUS])
    labels_b = [C.PRETTY.get(k, k) for k in vp.index]
    ax = axes[1]
    yb = np.arange(len(vp))
    ax.barh(yb, 1 - vp.values, color="#D5DBDB", label="Between sites")
    ax.barh(yb, vp.values, left=1 - vp.values, color="#C0392B",
            label="Within sites")
    for j, v in enumerate(vp.values):
        ax.text(1.01, j, f"{v*100:.1f}% within", va="center", fontsize=8)
    ax.set_yticks(yb)
    ax.set_yticklabels(labels_b)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Share of total variance (between-site + within-site)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.text(0.5, -0.12, "(B) Variance partition (continuous features)",
            transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Environmental features are fixed by collection site; only body length "
        f"varies within a site\n({meta['n_multi_sites']} multi-specimen sites, "
        f"{meta['n_specimens']} specimens)", fontsize=12)
    _save(fig, outpath)
