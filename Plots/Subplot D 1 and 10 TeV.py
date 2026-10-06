import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =========================================================
# File paths
# =========================================================
base_dir = r"C:\Users\GRIM REAPER\Downloads\Plots"

exp_files = {
    "FCC-LLP1": os.path.join(base_dir, "D-0-FCC-LLP1 1TeV.csv"),
    "FCC-LLP2": os.path.join(base_dir, "D-0-FCC-LLP2 1TeV.csv"),
}

exp_files_10tev = {
    "FCC-LLP1 10TeV": os.path.join(base_dir, "D-0-FCC-LLP1 10TeV.csv"),
    "FCC-LLP2 10TeV": os.path.join(base_dir, "D-0-FCC-LLP2 10TeV.csv"),
}

colors = {
    "FCC-LLP1": "#0072B2",        # Cerulean
    "FCC-LLP2": "#D55E00",       # Vermilion/Orange
    "FCC-LLP1 10TeV": "#0072B2",
    "FCC-LLP2 10TeV": "#D55E00",
}

Nthr = 3.0
x_right_boundary = 1.0
x_left_boundary = 0.10
y_top_boundary = 1e-4


# =========================================================
# Manual tick values
# =========================================================
x_ticks = [ 0.2, 0.4, 0.6, 0.8, 1.0]
y_ticks = [1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]

# Optional custom labels
# x_ticklabels = [r"0.1", r"0.3", r"0.5", r"0.7", r"0.9", r"1.1"]
# y_ticklabels = [r"$10^{-12}$", r"$10^{-10}$", r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"]


# =========================================================
# Robust parser
# =========================================================
def read_scan_file(fname):
    with open(fname, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = raw.replace("\ufeff", "")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    data_rows = []
    number_pat = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?')

    for ln in lines:
        if not re.search(r'\d', ln):
            continue

        vals = number_pat.findall(ln)
        if len(vals) < 8:
            continue

        nums = [float(v) for v in vals]

        if len(nums) >= 10:
            mN, Ve2, total_width, br_vis, ctau, nevents, brlnv, brlnc, nlnv, nlnc = nums[:10]
        elif len(nums) == 9:
            mN, Ve2, total_width, br_vis, ctau, brlnv, brlnc, nlnv, nlnc = nums[:9]
            nevents = np.nan
        elif len(nums) == 8:
            mN, Ve2, total_width, br_vis, ctau, brlnv, brlnc, nlnv = nums[:8]
            nevents = np.nan
            nlnc = np.nan
        else:
            continue

        data_rows.append([mN, Ve2, total_width, br_vis, ctau, nevents, brlnv, brlnc, nlnv, nlnc])

    df = pd.DataFrame(
        data_rows,
        columns=["mN", "Ve2", "TotalWidth", "BRVisible", "ctau", "Nevents", "BRLNV", "BRLNC", "NLNV", "NLNC"]
    )

    return df.sort_values(["mN", "Ve2"]).reset_index(drop=True)


# =========================================================
# Grid with zero extension
# =========================================================
def get_grid_with_zero_extension(df, zcol, x_extend=None):
    pivot = (
        df.pivot_table(index="Ve2", columns="mN", values=zcol, aggfunc="mean")
          .sort_index(axis=0)
          .sort_index(axis=1)
    )

    X = pivot.columns.values.astype(float)
    Y = pivot.index.values.astype(float)
    Z = pivot.values.astype(float)

    Z = np.ma.masked_invalid(Z)

    if x_extend is not None and x_extend > X.max():
        X = np.append(X, x_extend)
        zpad = np.zeros((Z.shape[0], 1), dtype=float)
        zpad = np.ma.array(zpad, mask=np.zeros_like(zpad, dtype=bool))
        Z = np.ma.hstack([Z, zpad])

    XX, YY = np.meshgrid(X, Y)
    return XX, YY, Z


# =========================================================
# Plot
# =========================================================
plt.rcParams.update({
    "font.size": 22,
    "axes.labelsize": 26,
    "axes.titlesize": 24,
    "legend.fontsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.linewidth": 1.8,
})

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
ax_lnv, ax_lnc = axes

all_mN = []
all_Ve2 = []


# -----------------------------
# Solid contours: baseline set
# -----------------------------
for exp_name, fname in exp_files.items():
    df = read_scan_file(fname)

    all_mN.extend(df["mN"].tolist())
    all_Ve2.extend(df["Ve2"].tolist())

    XX, YY, ZLNV = get_grid_with_zero_extension(df, "NLNV", x_extend=x_right_boundary)
    ax_lnv.contour(
        XX, YY, ZLNV,
        levels=[Nthr],
        colors=[colors[exp_name]],
        linewidths=4.0,
        linestyles="-"
    )

    XX, YY, ZLNC = get_grid_with_zero_extension(df, "NLNC", x_extend=x_right_boundary)
    ax_lnc.contour(
        XX, YY, ZLNC,
        levels=[Nthr],
        colors=[colors[exp_name]],
        linewidths=4.0,
        linestyles="-"
    )


# --------------------------------
# Dashed contours: 10 TeV set
# --------------------------------
for exp_name, fname in exp_files_10tev.items():
    df = read_scan_file(fname)

    all_mN.extend(df["mN"].tolist())
    all_Ve2.extend(df["Ve2"].tolist())

    XX, YY, ZLNV = get_grid_with_zero_extension(df, "NLNV", x_extend=x_right_boundary)
    ax_lnv.contour(
        XX, YY, ZLNV,
        levels=[Nthr],
        colors=[colors[exp_name]],
        linewidths=4.0,
        linestyles="--"
    )

    XX, YY, ZLNC = get_grid_with_zero_extension(df, "NLNC", x_extend=x_right_boundary)
    ax_lnc.contour(
        XX, YY, ZLNC,
        levels=[Nthr],
        colors=[colors[exp_name]],
        linewidths=4.0,
        linestyles="--"
    )


xmin = max(1e-3, np.nanmin(all_mN))
y_positive = [v for v in all_Ve2 if v > 0]
ymin = np.nanmin(y_positive)
ymax = np.nanmax(y_positive)


for ax, title in zip(
    axes,
    [r"$D^0 \to N N$ (LNV)", r"$D^0 \to N N$ (LNC)"]
):
    ax.set_yscale("log")
    ax.set_xlim(x_left_boundary, x_right_boundary)
    ax.set_ylim(ymin, y_top_boundary)
    ax.set_xlabel(r"$m_N \,[\mathrm{GeV}]$")
    ax.set_title(title, pad=12)

    # Manual tick positions
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Optional manual labels
    # ax.set_xticklabels(x_ticklabels)
    # ax.set_yticklabels(y_ticklabels)

    ax.tick_params(axis="both", which="major", direction="in", length=8, width=1.6, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.2, top=True, right=True)

    # Turn off automatic minor ticks so only chosen y ticks are shown
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.grid(which="major", linestyle="--", linewidth=1, alpha=0.5)


ax_lnv.set_ylabel(r"$|V_{eN}|^2$")


legend_handles = [
    plt.Line2D([0], [0], color="#0072B2", lw=4, linestyle="-", label="FACET [$\Lambda=1$ TeV]"),
    plt.Line2D([0], [0], color="#D55E00", lw=4, linestyle="-", label="FASER2 [$\Lambda=1$ TeV]"),
    plt.Line2D([0], [0], color="#009E73", lw=4, linestyle="-", label="SHiP [$\Lambda=1$ TeV]"),
    plt.Line2D([0], [0], color="#0072B2", lw=4, linestyle="--", label="FACET [$\Lambda=10$ TeV]"),
    plt.Line2D([0], [0], color="#D55E00", lw=4, linestyle="--", label="FASER2 [$\Lambda=10$ TeV]"),
    plt.Line2D([0], [0], color="#009E73", lw=4, linestyle="--", label="SHiP [$\Lambda=10$ TeV]"),
]

ax_lnv.legend(handles=legend_handles, loc="lower left", frameon=True, fontsize=14)

plt.tight_layout()
plt.show()