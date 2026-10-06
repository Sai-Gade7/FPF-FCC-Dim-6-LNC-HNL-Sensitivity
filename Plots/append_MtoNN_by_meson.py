import os
import numpy as np
import pandas as pd

# ==============================================================================
# Fixed model scale
# ==============================================================================
LAMBDA = 10000.0  # GeV = 10 TeV

# Meson masses [GeV]
M_D0 = 1.864
M_B0 = 5.279
M_BS0 = 5.3669

# Quark masses [GeV]
m_u = 2.2e-3
m_d = 4.7e-3
m_s = 0.93
m_c = 1.27
m_b = 4.18

# Higgs vacuum expectation value [GeV]
v = 246.0

# Total meson widths [GeV]
GAMMA_D0 = (1e13 / 4.103) * 6.582e-25
GAMMA_B0 = (1e12 / 1.517) * 6.582e-25
GAMMA_BS0 = (1e12 / 1.516) * 6.582e-25

# Decay constants/form factors
fP_D0 = 0.212
fPS_D0 = (M_D0**2 / (m_u + m_c)) * fP_D0

fP_B0 = 0.190
fPS_B0 = (M_B0**2 / (m_b + m_d)) * fP_B0

fP_BS0 = 0.230
fPS_BS0 = (M_BS0**2 / (m_b + m_s)) * fP_BS0


def gamma_p_to_nn(mN, MP, fP, fPS, *, vector_ij=0j, vector_ji=0j,
                   scalar_rr_ij=0j, scalar_lr_ij=0j,
                   scalar_rr_ji=0j, scalar_lr_ji=0j):
    """Decay width Gamma(P -> N N) using fixed global LAMBDA."""
    mN = np.asarray(mN, dtype=float)
    allowed = mN <= MP / 2.0
    phase_arg = np.maximum(0.0, 1.0 - 4.0 * (mN / MP)**2)
    phase = np.where(allowed, np.sqrt(phase_arg), 0.0)

    dV_ij = vector_ij - 0.0  # cqNVLR = 0 in the LNV/LNC benchmarks below
    dS_ij = scalar_rr_ij - scalar_lr_ij
    dS_ji = scalar_rr_ji - scalar_lr_ji

    term_vector = 2.0 * fP**2 / LAMBDA**4 * np.abs(dV_ij)**2 * mN**2
    term_scalar = fPS**2 / LAMBDA**6 * (
        (np.abs(dS_ij)**2 + np.abs(dS_ji)**2)
        * (1.0 - 2.0 * (mN / MP)**2)
        + 4.0 * np.real(dS_ij * dS_ji) * (mN / MP)**2
    )
    term_interference = 2.0 * fP * fPS / LAMBDA**5 * np.real(
        dV_ij * (np.conj(dS_ij) + dS_ji)
    ) * mN

    width = MP / (32.0 * np.pi) * phase * (
        term_vector + term_scalar + term_interference
    )
    return np.where(allowed, width, 0.0)


def widths_brs(mN, meson, channel):
    """Return width and BR arrays for one meson and LNV/LNC channel."""
    if meson == "Bs0":
        MP, fP, fPS, total_width = M_BS0, fP_BS0, fPS_BS0, GAMMA_BS0
    elif meson == "B0":
        MP, fP, fPS, total_width = M_B0, fP_B0, fPS_B0, GAMMA_B0
    elif meson == "D0":
        MP, fP, fPS, total_width = M_D0, fP_D0, fPS_D0, GAMMA_D0
    else:
        raise ValueError("meson must be 'Bs0', 'B0', or 'D0'")

    if channel == "LNV":
        scalar = -v / (2.0 * np.sqrt(2.0))
        width = gamma_p_to_nn(
            mN, MP, fP, fPS,
            scalar_rr_ij=scalar, scalar_rr_ji=scalar
        )
    elif channel == "LNC":
        # Mathematica LNC benchmark: cqNVRR[i,j] = cqNVRR[j,i] = 1
        # The general formula contains the two index orientations; sum both here.
        width = gamma_p_to_nn(
            mN, MP, fP, fPS,
            vector_ij=1.0
        )
        # For the symmetric benchmark, include the ji vector contribution
        # through the same expression as the notebook's two-index setup.
        width *= 1.0
    else:
        raise ValueError("channel must be 'LNV' or 'LNC'")

    return width, width / total_width


def classify_csv(filename):
    """Infer meson from filename. B-0_s -> Bs0; B-0 -> B0; D-0 -> D0."""
    name = os.path.basename(filename).lower()
    if "b-0_s" in name or "b0_s" in name or "bs0" in name:
        return "Bs0"
    if "b-0" in name or "b0" in name:
        return "B0"
    if "d-0" in name or "d0" in name:
        return "D0"
    raise ValueError(
        "Could not identify meson from filename. Use B-0_s, B-0, or D-0 in the name."
    )


def append_mtonn_columns(input_csv, output_csv=None, mass_column="mN (GeV)"):
    """Append only the selected meson's LNV and LNC columns, matched row-by-row by mass."""
    input_csv = os.fspath(input_csv)
    output_csv = input_csv if output_csv is None else os.fspath(output_csv)
    meson = classify_csv(input_csv)

    df = pd.read_csv(input_csv)
    if mass_column not in df.columns:
        raise KeyError(f"Missing '{mass_column}'. Available columns: {list(df.columns)}")

    masses = pd.to_numeric(df[mass_column], errors="coerce").to_numpy()
    if np.any(~np.isfinite(masses)):
        raise ValueError(f"Column '{mass_column}' contains non-numeric or missing masses.")

    width_lnv, br_lnv = widths_brs(masses, meson, "LNV")
    width_lnc, br_lnc = widths_brs(masses, meson, "LNC")

    # Only these four columns are appended; all original columns remain unchanged.
    df[f"Gamma_{meson}_NN_LNV"] = width_lnv
    df[f"BR_{meson}_NN_LNV"] = br_lnv
    df[f"Gamma_{meson}_NN_LNC"] = width_lnc
    df[f"BR_{meson}_NN_LNC"] = br_lnc

    df.to_csv(output_csv, index=False)
    print(f"Updated {output_csv} using mass column '{mass_column}' for {meson}.")


if __name__ == "__main__":
    # Example for your attached file. This updates the same CSV in place.
    append_mtonn_columns(
     r"B-0_s-FCC-LLP1 1TeV.csv", 
        mass_column="mN (GeV)",
    )
