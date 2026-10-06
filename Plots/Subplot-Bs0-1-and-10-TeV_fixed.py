import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =========================================================
# File paths
# =========================================================
base_dir = r"D:\2026 HNL\FCC HNL FORESEE\Plots"

exp_files = {
    "FCC-LLP1": os.path.join(base_dir, "B-0_s-FCC-LLP1 1TeV.csv"),
    "FCC-LLP2": os.path.join(base_dir, "B-0_s-FCC-LLP2 1TeV.csv"),
}

exp_files_10tev = {
    "FCC-LLP1 10TeV": os.path.join(base_dir, "B-0_s-FCC-LLP1 10TeV.csv"),
    "FCC-LLP2 10TeV": os.path.join(base_dir, "B-0_s-FCC-LLP2 10TeV.csv"),
}

colors = {
    "FCC-LLP1": plt.cm.hsv(0.53),
    "FCC-LLP2": plt.cm.hsv(0.93),
    "FCC-LLP1 10TeV": plt.cm.hsv(0.53),
    "FCC-LLP2 10TeV": plt.cm.hsv(0.93),
}

Nthr = 3.0
x_right_boundary = 3.0
x_left_boundary = 0.10
y_top_boundary = 1e-2


# =========================================================
# Manual tick values
# =========================================================
x_ticks = [ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y_ticks = [1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

# Optional custom labels
# x_ticklabels = [r"0.1", r"0.3", r"0.5", r"0.7", r"0.9", r"1.1"]
# y_ticklabels = [r"$10^{-12}$", r"$10^{-10}$", r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"]


# =========================================================
# Parser for appended CSV format
# =========================================================
def read_scan_file(fname):
    df = pd.read_csv(fname)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "mN (GeV)": "mN",
        "Ve^2": "Ve2",
        "Total Width": "TotalWidth",
        "BR Visible": "BRVisible",
        "ctau (m)": "ctau",
        "N_events": "Nevents",
        "Gamma_Bs0_NN_LNV": "GammaLNV",
        "BR_Bs0_NN_LNV": "BRLNV",
        "Gamma_Bs0_NN_LNC": "GammaLNC",
        "BR_Bs0_NN_LNC": "BRLNC",
    }
    df = df.rename(columns=rename_map)

    required = ["mN", "Ve2", "TotalWidth", "BRVisible", "ctau", "Nevents", "BRLNV", "BRLNC"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {fname}: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["mN", "Ve2", "Nevents", "BRLNV", "BRLNC"]).copy()

    df["NLNV"] = df["Nevents"] * df["BRLNV"]
    df["NLNC"] = df["Nevents"] * df["BRLNC"]

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
    [r"$B^0_{s} \to N N$ (LNV)", r"$B^0_{s} \to N N$ (LNC)"]
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
    plt.Line2D([0], [0], color=plt.cm.hsv(0.53),  lw=4, linestyle="-",  label=r"FCC-LLP1 [$\Lambda=1$ TeV]"),
    plt.Line2D([0], [0], color=plt.cm.hsv(0.93), lw=4, linestyle="-",  label=r"FCC-LLP2 [$\Lambda=1$ TeV]"),
    plt.Line2D([0], [0], color=plt.cm.hsv(0.53),  lw=4, linestyle="--", label=r"FCC-LLP1 [$\Lambda=10$ TeV]"),
    plt.Line2D([0], [0], color=plt.cm.hsv(0.93), lw=4, linestyle="--", label=r"FCC-LLP2 [$\Lambda=10$ TeV]"),
]


ax_lnv.legend(handles=legend_handles, loc="lower left", frameon=True, fontsize=14)

plt.tight_layout()
plt.show()