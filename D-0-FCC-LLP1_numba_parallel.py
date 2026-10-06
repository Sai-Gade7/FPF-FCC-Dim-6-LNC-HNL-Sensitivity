import numpy as np
import pandas as pd
from numba import njit, prange

mD0 = 1.864
mB0 = 5.279
mBs0 = 5.3669
mpi = 0.139570
mk = 0.493677
mko = 0.498
mpo = 0.1349766
meta = 0.547853
mrho = 0.77511
mrhoo = 0.77549
mw = 0.782650
me = 0.000511
mmu = 0.105658
mkstar = 0.89166
mtau = 1.77686
mD = 1.86961
mDs = 1.96830
metaprime = 0.95758
metac = 2.9836
mDstar = 2.01027
mDSstar = 2.1121
mphi = 1.019461
mjpsi = 3.096916
mB = 5.2791
mBstar = 5.3251
mBc = 6.275
mnu = 0.0

mu_quark = 2.16e-3
md = 4.67e-3
ms = 93e-3
mc = 1.27
mb = 4.18
mt = 172.76

fB = 0.188
fBc = 0.436
fBstar = 0.175
fD = 0.2037
fDs = 0.2578
fetaprime = 0.1529
fetac = 0.335
fDstar = 0.310
fDSstar = 0.315
fphi = 0.229
fjpsi = 0.459
fpi = 0.13050
fk = 0.15572
fpo = 0.135
feta = 0.1647
fko = 0.1598
frho = 0.220
frhoo = 0.220
fw = 0.195
fkstar = 0.217

Vud = 0.97425
Vus = 0.2253
Vcd = 0.225
Vcs = 0.986
Vub = 4.41e-3
Vcb = 40.8e-3
VCKM = np.array([[Vud, Vus, Vub], [Vcd, Vcs, Vcb]], dtype=np.float64)

mup = np.array([mu_quark, mc, mt], dtype=np.float64)
mdown = np.array([md, ms, mb], dtype=np.float64)

GF = 1.166e-5
MWL = 80.385
mh = 125.0
g = np.sqrt(8 * MWL**2 * GF / np.sqrt(2))
MZL = 91.1876
xw = np.sqrt(0.231)

gL = -(1 / 2) + xw**2
gR = xw**2
gLu = 1 / 2 - (2 / 3) * xw**2
gRu = -(2 / 3) * xw**2
gLd = -(1 / 2) + (1 / 3) * xw**2
gRd = (1 / 3) * xw**2

Kpi = -(1 / (2 * np.sqrt(2)))
Keta = -(1 / (2 * np.sqrt(6)))
Ketap = 1 / (4 * np.sqrt(3))
Ketac = -(1 / 4)
Krho = (1 / np.sqrt(2)) * (1 / 2 - xw**2)
Komega = -(1 / (3 * np.sqrt(2))) * xw**2
Kphi = (-(1 / 4) + (1 / 3) * xw**2)
KK0 = 1 / 4
KK0s = (-(1 / 4) + (1 / 3) * xw**2)
Kjpsi = (1 / 4 - (2 / 3) * xw**2)

CA = 3
vev = 246.0
alpha_em = 1 / 137
cw = np.sqrt(1 - xw**2)
mu0 = 0.957

mP = mD0
L1 = 1500.0
L2 = 1550.0
Rdet = 2.5
thetaM = 0.0
INTEGRAL_STEPS = 400

@njit(cache=True)
def kallen_lambda(x, y, z):
    val = x * x + y * y + z * z - 2 * x * y - 2 * x * z - 2 * y * z
    if val < 0.0:
        val = 0.0
    return np.sqrt(val)

@njit(cache=True)
def Fp(x, y):
    if 1.0 - x - y < 0.0:
        return 0.0
    lam = kallen_lambda(1.0, x * x, y * y)
    return ((1.0 + x * x) * (1.0 + x * x - y * y) - 4.0 * x * x) * lam

@njit(cache=True)
def Fv(x, y):
    if 1.0 - x - y < 0.0:
        return 0.0
    lam = kallen_lambda(1.0, x * x, y * y)
    return ((1.0 - x * x) ** 2 + (1.0 + x * x) * y * y - 2.0 * y ** 4) * lam

@njit(cache=True)
def _integrand_I1(s, x, y, z):
    lam1 = kallen_lambda(s, x * x, y * y)
    lam2 = kallen_lambda(1.0, s, z * z)
    return (1.0 / s) * (s - x * x - y * y) * (1.0 + z * z - s) * lam1 * lam2

@njit(cache=True)
def _integrand_I2(s, x, y, z):
    lam1 = kallen_lambda(s, y * y, z * z)
    lam2 = kallen_lambda(1.0, s, x * x)
    return (1.0 / s) * (1.0 + x * x - s) * lam1 * lam2

@njit(cache=True)
def _simpson_I1(x, y, z, nsteps):
    lower = (x + y) ** 2
    upper = (1.0 - z) ** 2
    if upper <= lower:
        return 0.0
    if nsteps % 2 == 1:
        nsteps += 1
    h = (upper - lower) / nsteps
    total = _integrand_I1(lower, x, y, z) + _integrand_I1(upper, x, y, z)
    for i in range(1, nsteps):
        s = lower + i * h
        if i % 2 == 0:
            total += 2.0 * _integrand_I1(s, x, y, z)
        else:
            total += 4.0 * _integrand_I1(s, x, y, z)
    return total * h / 3.0

@njit(cache=True)
def _simpson_I2(x, y, z, nsteps):
    lower = (y + z) ** 2
    upper = (1.0 - x) ** 2
    if upper <= lower:
        return 0.0
    if nsteps % 2 == 1:
        nsteps += 1
    h = (upper - lower) / nsteps
    total = _integrand_I2(lower, x, y, z) + _integrand_I2(upper, x, y, z)
    for i in range(1, nsteps):
        s = lower + i * h
        if i % 2 == 0:
            total += 2.0 * _integrand_I2(s, x, y, z)
        else:
            total += 4.0 * _integrand_I2(s, x, y, z)
    return total * h / 3.0

@njit(cache=True)
def I1(x, y, z):
    if 1.0 - x - y - z < 0.0:
        return 0.0
    return _simpson_I1(x, y, z, INTEGRAL_STEPS)

@njit(cache=True)
def I2(x, y, z):
    if 1.0 - x - y - z < 0.0:
        return 0.0
    return y * z * _simpson_I2(x, y, z, INTEGRAL_STEPS)

@njit(cache=True)
def Gamma_N_to_l_pi(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fpi * fpi * Vud * Vud
        terms = Ve * Ve * Fp(me / mN, mpi / mN) + Vmu * Vmu * Fp(mmu / mN, mpi / mN) + Vtau * Vtau * Fp(mtau / mN, mpi / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_K(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fk * fk * Vus * Vus
        terms = Ve * Ve * Fp(me / mN, mk / mN) + Vmu * Vmu * Fp(mmu / mN, mk / mN) + Vtau * Vtau * Fp(mtau / mN, mk / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_D(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fD * fD * Vcd * Vcd
        terms = Ve * Ve * Fp(me / mN, mD / mN) + Vmu * Vmu * Fp(mmu / mN, mD / mN) + Vtau * Vtau * Fp(mtau / mN, mD / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_Ds(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fDs * fDs * Vcs * Vcs
        terms = Ve * Ve * Fp(me / mN, mDs / mN) + Vmu * Vmu * Fp(mmu / mN, mDs / mN) + Vtau * Vtau * Fp(mtau / mN, mDs / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_B(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fB * fB * Vub * Vub
        terms = Ve * Ve * Fp(me / mN, mB / mN) + Vmu * Vmu * Fp(mmu / mN, mB / mN) + Vtau * Vtau * Fp(mtau / mN, mB / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_Bc(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fBc * fBc * Vcb * Vcb
        terms = Ve * Ve * Fp(me / mN, mBc / mN) + Vmu * Vmu * Fp(mmu / mN, mBc / mN) + Vtau * Vtau * Fp(mtau / mN, mBc / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_P(mN, Ve, Vmu, Vtau):
    return Gamma_N_to_l_pi(mN, Ve, Vmu, Vtau) + Gamma_N_to_l_K(mN, Ve, Vmu, Vtau) + Gamma_N_to_l_D(mN, Ve, Vmu, Vtau) + Gamma_N_to_l_Ds(mN, Ve, Vmu, Vtau) + Gamma_N_to_l_B(mN, Ve, Vmu, Vtau) + Gamma_N_to_l_Bc(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def Gamma_N_to_l_rho(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * frho * frho * Vud * Vud
        terms = Ve * Ve * Fv(me / mN, mrho / mN) + Vmu * Vmu * Fv(mmu / mN, mrho / mN) + Vtau * Vtau * Fv(mtau / mN, mrho / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_Kstar(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fkstar * fkstar * Vus * Vus
        terms = Ve * Ve * Fv(me / mN, mkstar / mN) + Vmu * Vmu * Fv(mmu / mN, mkstar / mN) + Vtau * Vtau * Fv(mtau / mN, mkstar / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_Dstar(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fDstar * fDstar * Vcd * Vcd
        terms = Ve * Ve * Fv(me / mN, mDstar / mN) + Vmu * Vmu * Fv(mmu / mN, mDstar / mN) + Vtau * Vtau * Fv(mtau / mN, mDstar / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_DSstar(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fDSstar * fDSstar * Vcs * Vcs
        terms = Ve * Ve * Fv(me / mN, mDSstar / mN) + Vmu * Vmu * Fv(mmu / mN, mDSstar / mN) + Vtau * Vtau * Fv(mtau / mN, mDSstar / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_Bstar(mN, Ve, Vmu, Vtau):
    if mN <= MWL:
        coeff = (GF * GF * mN ** 3) / (16.0 * np.pi) * fBstar * fBstar * Vub * Vub
        terms = Ve * Ve * Fv(me / mN, mBstar / mN) + Vmu * Vmu * Fv(mmu / mN, mBstar / mN) + Vtau * Vtau * Fv(mtau / mN, mBstar / mN)
        return coeff * terms
    return 0.0

@njit(cache=True)
def Gamma_N_to_l_V(mN, Ve, Vmu, Vtau):
    return Gamma_N_to_l_rho(mN, Ve, Vmu, Vtau) + Gamma_N_to_l_Kstar(mN, Ve, Vmu, Vtau) + Gamma_N_to_l_Dstar(mN, Ve, Vmu, Vtau) + Gamma_N_to_l_DSstar(mN, Ve, Vmu, Vtau) + Gamma_N_to_l_Bstar(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def Gamma_N_to_e_mu_nu(mN, Ve):
    if mN <= MWL:
        return (GF * GF * mN ** 5) / (16.0 * np.pi ** 3) * Ve * Ve * I1(me / mN, mnu / mN, mmu / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_mu_e_nu(mN, Vmu):
    if mN <= MWL:
        return (GF * GF * mN ** 5) / (16.0 * np.pi ** 3) * Vmu * Vmu * I1(mmu / mN, mnu / mN, me / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_e_tau_nu(mN, Ve):
    if mN <= MWL:
        return (GF * GF * mN ** 5) / (16.0 * np.pi ** 3) * Ve * Ve * I1(me / mN, mnu / mN, mtau / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_tau_e_nu(mN, Vtau):
    if mN <= MWL:
        return (GF * GF * mN ** 5) / (16.0 * np.pi ** 3) * Vtau * Vtau * I1(mtau / mN, mnu / mN, me / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_mu_tau_nu(mN, Vmu):
    if mN <= MWL:
        return (GF * GF * mN ** 5) / (16.0 * np.pi ** 3) * Vmu * Vmu * I1(mmu / mN, mnu / mN, mtau / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_tau_mu_nu(mN, Vtau):
    if mN <= MWL:
        return (GF * GF * mN ** 5) / (16.0 * np.pi ** 3) * Vtau * Vtau * I1(mtau / mN, mnu / mN, mmu / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_l1l2_nul2(mN, Ve, Vmu, Vtau):
    return Gamma_N_to_e_mu_nu(mN, Ve) + Gamma_N_to_mu_e_nu(mN, Vmu) + Gamma_N_to_e_tau_nu(mN, Ve) + Gamma_N_to_tau_e_nu(mN, Vtau) + Gamma_N_to_mu_tau_nu(mN, Vmu) + Gamma_N_to_tau_mu_nu(mN, Vtau)

@njit(cache=True)
def Gamma_N_to_nu_pi0(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * Kpi * Kpi * fpo * fpo * Fp(mnu / mN, mpo / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_eta(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * Keta * Keta * feta * feta * Fp(mnu / mN, meta / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_K0(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * KK0 * KK0 * fko * fko * Fp(mnu / mN, mko / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_etaprime(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * Ketap * Ketap * fetaprime * fetaprime * Fp(mnu / mN, metaprime / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_etac(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * Ketac * Ketac * fetac * fetac * Fp(mnu / mN, metac / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_P(mN, Ve, Vmu, Vtau):
    return Gamma_N_to_nu_pi0(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_eta(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_etaprime(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_etac(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_K0(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def Gamma_N_to_nu_rho0(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * frhoo * frhoo * Krho * Krho * Fv(mnu / mN, mrhoo / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_omega(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * fw * fw * Komega * Komega * Fv(mnu / mN, mw / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_phi(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * fphi * fphi * Kphi * Kphi * Fv(mnu / mN, mphi / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_Ks(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * fkstar * fkstar * KK0s * KK0s * Fv(mnu / mN, mkstar / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_Jpsi(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 3) / (4.0 * np.pi)
        mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
        return coeff * mix * fjpsi * fjpsi * Kjpsi * Kjpsi * Fv(mnu / mN, mjpsi / mN)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_V(mN, Ve, Vmu, Vtau):
    return Gamma_N_to_nu_rho0(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_omega(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_phi(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_Jpsi(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_Ks(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def Gamma_N_to_nu_eee(mN, Ve):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3)
        term = ((gL * gL + gR * gR + 1.0 + 2.0 * gL) * I1(mnu / mN, me / mN, me / mN) + 2.0 * gR * (gL + 1.0) * I2(mnu / mN, me / mN, me / mN))
        return coeff * Ve * Ve * term
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_muee(mN, Vmu):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3)
        term = ((gL * gL + gR * gR) * I1(mnu / mN, me / mN, me / mN) + 2.0 * gL * gR * I2(mnu / mN, me / mN, me / mN))
        return coeff * Vmu * Vmu * term
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_tauee(mN, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3)
        term = ((gL * gL + gR * gR) * I1(mnu / mN, me / mN, me / mN) + 2.0 * gL * gR * I2(mnu / mN, me / mN, me / mN))
        return coeff * Vtau * Vtau * term
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_emu_mu(mN, Ve):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3)
        term = ((gL * gL + gR * gR) * I1(mnu / mN, mmu / mN, mmu / mN) + 2.0 * gL * gR * I2(mnu / mN, mmu / mN, mmu / mN))
        return coeff * Ve * Ve * term
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_mumu_mu(mN, Vmu):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3)
        term = ((gL * gL + gR * gR + 1.0 + 2.0 * gL) * I1(mnu / mN, mmu / mN, mmu / mN) + 2.0 * gR * (gL + 1.0) * I2(mnu / mN, mmu / mN, mmu / mN))
        return coeff * Vmu * Vmu * term
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_taumu_mu(mN, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3)
        term = ((gL * gL + gR * gR) * I1(mnu / mN, mmu / mN, mmu / mN) + 2.0 * gL * gR * I2(mnu / mN, mmu / mN, mmu / mN))
        return coeff * Vtau * Vtau * term
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_etau_tau(mN, Ve):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3)
        term = ((gL * gL + gR * gR) * I1(mnu / mN, mtau / mN, mtau / mN) + 2.0 * gL * gR * I2(mnu / mN, mtau / mN, mtau / mN))
        return coeff * Ve * Ve * term
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_mutau_tau(mN, Vmu):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3)
        term = ((gL * gL + gR * gR) * I1(mnu / mN, mtau / mN, mtau / mN) + 2.0 * gL * gR * I2(mnu / mN, mtau / mN, mtau / mN))
        return coeff * Vmu * Vmu * term
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_tautau_tau(mN, Vtau):
    if mN <= MZL:
        coeff = 2.0 * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3)
        term = ((gL * gL + gR * gR + 1.0 + 2.0 * gL) * I1(mnu / mN, mtau / mN, mtau / mN) + 2.0 * gR * (gL + 1.0) * I2(mnu / mN, mtau / mN, mtau / mN))
        return coeff * Vtau * Vtau * term
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_l1l2l2(mN, Ve, Vmu, Vtau):
    return Gamma_N_to_nu_eee(mN, Ve) + Gamma_N_to_nu_muee(mN, Vmu) + Gamma_N_to_nu_tauee(mN, Vtau) + Gamma_N_to_nu_emu_mu(mN, Ve) + Gamma_N_to_nu_mumu_mu(mN, Vmu) + Gamma_N_to_nu_taumu_mu(mN, Vtau) + Gamma_N_to_nu_etau_tau(mN, Ve) + Gamma_N_to_nu_mutau_tau(mN, Vmu) + Gamma_N_to_nu_tautau_tau(mN, Vtau)

@njit(cache=True)
def Gamma_N_to_3nu(mN, Ve, Vmu, Vtau):
    if mN <= MZL:
        return (2.0 * GF * GF * mN ** 5) / (192.0 * np.pi ** 3) * (Ve * Ve + Vmu * Vmu + Vtau * Vtau)
    return 0.0

@njit(cache=True)
def Gamma_N_to_ljj(mN, Ve, Vmu, Vtau):
    if mN > MWL:
        return 0.0
    total_e = 0.0
    total_mu = 0.0
    total_tau = 0.0
    for i in range(2):
        for j in range(3):
            up = mup[i]
            down = mdown[j]
            vckm = VCKM[i, j]
            if mN > me + up + down:
                total_e += vckm * vckm * I1(me / mN, up / mN, down / mN)
            if mN > mmu + up + down:
                total_mu += vckm * vckm * I1(mmu / mN, up / mN, down / mN)
            if mN > mtau + up + down:
                total_tau += vckm * vckm * I1(mtau / mN, up / mN, down / mN)
    return CA * (GF * GF * mN ** 5) / (16.0 * np.pi ** 3) * (Ve * Ve * total_e + Vmu * Vmu * total_mu + Vtau * Vtau * total_tau)

@njit(cache=True)
def Gamma_N_to_nu_jj(mN, Ve, Vmu, Vtau):
    if mN > MZL:
        return 0.0
    total_u = 0.0
    total_d = 0.0
    for i in range(2):
        q = mup[i]
        if mN > 2.0 * q:
            total_u += ((gLu * gLu + gRu * gRu) * I1(mnu / mN, q / mN, q / mN) + 2.0 * gLu * gRu * I2(mnu / mN, q / mN, q / mN))
    for i in range(3):
        q = mdown[i]
        if mN > 2.0 * q:
            total_d += ((gLd * gLd + gRd * gRd) * I1(mnu / mN, q / mN, q / mN) + 2.0 * gLd * gRd * I2(mnu / mN, q / mN, q / mN))
    mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
    return CA * (GF * GF * mN ** 5) / (8.0 * np.pi ** 3) * mix * (total_u + total_d)

@njit(cache=True)
def Gamma_N_lW(mN, Ve, Vmu, Vtau):
    mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
    if mN > MWL:
        return mix * g * g / (64.0 * np.pi) * ((mN * mN - MWL * MWL) ** 2 * (mN * mN + 2.0 * MWL * MWL)) / (mN ** 3 * MWL * MWL)
    return 0.0

@njit(cache=True)
def Gamma_N_nuZ(mN, Ve, Vmu, Vtau):
    mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
    if mN > MZL:
        return mix * g * g / (64.0 * np.pi * cw * cw) * ((mN * mN - MZL * MZL) ** 2 * (mN * mN + 2.0 * MZL * MZL)) / (mN ** 3 * MZL * MZL)
    return 0.0

@njit(cache=True)
def Gamma_N_nuH(mN, Ve, Vmu, Vtau):
    mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
    if mN > mh:
        return mix * (mN * mN - mh * mh) ** 2 / ((16.0 * np.pi * mN) * vev * vev)
    return 0.0

@njit(cache=True)
def Gamma_N_to_nu_gamma(mN, Ve, Vmu, Vtau):
    mix = Ve * Ve + Vmu * Vmu + Vtau * Vtau
    return (mN ** 3 * mmu * mmu * alpha_em) / (128.0 * np.pi ** 6 * vev ** 4) * mix

@njit(cache=True)
def Gamma_had(mN, Ve, Vmu, Vtau):
    if mN < mu0:
        return 2.0 * Gamma_N_to_l_P(mN, Ve, Vmu, Vtau) + 2.0 * Gamma_N_to_l_V(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_P(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_V(mN, Ve, Vmu, Vtau)
    return Gamma_N_to_nu_jj(mN, Ve, Vmu, Vtau) + 2.0 * Gamma_N_to_ljj(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def Gamma_N_total(mN, Ve, Vmu, Vtau):
    return 2.0 * Gamma_N_to_l1l2_nul2(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_l1l2l2(mN, Ve, Vmu, Vtau) + Gamma_N_to_3nu(mN, Ve, Vmu, Vtau) + Gamma_had(mN, Ve, Vmu, Vtau) + 2.0 * Gamma_N_lW(mN, Ve, Vmu, Vtau) + Gamma_N_nuZ(mN, Ve, Vmu, Vtau) + Gamma_N_nuH(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_gamma(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def LN(mN, Ve, Vmu, Vtau):
    return 0.197e-15 / Gamma_N_total(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def BRNtolX(mN, Ve, Vmu, Vtau):
    if mN < mu0:
        return (2.0 * Gamma_N_to_l_P(mN, Ve, Vmu, Vtau) + 2.0 * Gamma_N_to_l_V(mN, Ve, Vmu, Vtau)) / Gamma_N_total(mN, Ve, Vmu, Vtau)
    return (2.0 * Gamma_N_to_ljj(mN, Ve, Vmu, Vtau)) / Gamma_N_total(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def BRNto_nuX(mN, Ve, Vmu, Vtau):
    if mN < mu0:
        return (Gamma_N_to_nu_P(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_V(mN, Ve, Vmu, Vtau)) / Gamma_N_total(mN, Ve, Vmu, Vtau)
    return Gamma_N_to_nu_jj(mN, Ve, Vmu, Vtau) / Gamma_N_total(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def BRNto_llnu(mN, Ve, Vmu, Vtau):
    return (2.0 * Gamma_N_to_l1l2_nul2(mN, Ve, Vmu, Vtau) + Gamma_N_to_nu_l1l2l2(mN, Ve, Vmu, Vtau)) / Gamma_N_total(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def BRNto3nu(mN, Ve, Vmu, Vtau):
    return Gamma_N_to_3nu(mN, Ve, Vmu, Vtau) / Gamma_N_total(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def BRNto_vis(mN, Ve, Vmu, Vtau):
    return BRNtolX(mN, Ve, Vmu, Vtau) + BRNto_nuX(mN, Ve, Vmu, Vtau) + BRNto_llnu(mN, Ve, Vmu, Vtau)

@njit(cache=True)
def EP(PP):
    return np.sqrt(PP * PP + mP * mP)

@njit(cache=True)
def gammaP(PP):
    return EP(PP) / mP

@njit(cache=True)
def betaP(PP):
    return PP / EP(PP)

@njit(cache=True)
def gammaNBoost(mN, PP, theta):
    if mN >= mP / 2.0:
        return np.nan
    rootTerm = np.sqrt(1.0 - (4.0 * mN * mN) / (mP * mP))
    return gammaP(PP) * (mP / (2.0 * mN)) * (1.0 + betaP(PP) * rootTerm * np.cos(thetaM))

@njit(cache=True)
def ENBoost(mN, PP, theta):
    gN = gammaNBoost(mN, PP, theta)
    if np.isnan(gN):
        return np.nan
    return gN * mN

@njit(cache=True)
def pNBoost(mN, PP, theta):
    EN = ENBoost(mN, PP, theta)
    if np.isnan(EN) or EN < mN:
        return np.nan
    return np.sqrt(EN * EN - mN * mN)

@njit(cache=True)
def betaNBoost(mN, PP, theta):
    EN = ENBoost(mN, PP, theta)
    pN = pNBoost(mN, PP, theta)
    if np.isnan(EN) or np.isnan(pN) or EN == 0.0:
        return np.nan
    return pN / EN

@njit(cache=True)
def PDecay(mN, Ve, Vmu, Vtau, PP, theta):
    gN = gammaNBoost(mN, PP, theta)
    bN = betaNBoost(mN, PP, theta)
    ctau = LN(mN, Ve, Vmu, Vtau)
    if np.isnan(gN) or np.isnan(bN) or ctau <= 0.0:
        return np.nan
    dlab = bN * gN * ctau
    Delta = L2 - L1
    return (Delta / dlab) * np.exp(-L1 / dlab) * (1.0 if (Rdet - theta * L1) >= 0.0 else 0.0)

@njit(parallel=True, cache=True)
def compute_rows(mN_grid, Ve_grid, p_arr, theta_arr, NM_arr, valVMu, valVTau):
    n = mN_grid.size * Ve_grid.size
    out = np.empty((n, 6), dtype=np.float64)
    for idx in prange(n):
        i = idx // Ve_grid.size
        j = idx % Ve_grid.size
        mN = mN_grid[i]
        Ve = Ve_grid[j]
        totalWidth = Gamma_N_total(mN, Ve, valVMu, valVTau)
        brVis = BRNto_vis(mN, Ve, valVMu, valVTau)
        ctauN = LN(mN, Ve, valVMu, valVTau)
        nEvents = 0.0
        for k in range(p_arr.size):
            pdec = PDecay(mN, Ve, valVMu, valVTau, p_arr[k], theta_arr[k])
            if np.isfinite(pdec):
                nEvents += 2.0 * NM_arr[k] * brVis * pdec
        out[idx, 0] = mN
        out[idx, 1] = Ve * Ve
        out[idx, 2] = totalWidth
        out[idx, 3] = brVis
        out[idx, 4] = ctauN
        out[idx, 5] = nEvents
    return out

if __name__ == "__main__":
    df = pd.read_csv("combined_spectrum_pid_421 for FCC-LLP1.csv")
    p_arr = df["p"].to_numpy(np.float64)
    theta_arr = df["theta"].to_numpy(np.float64)
    NM_arr = df["N_M"].to_numpy(np.float64)

    valVMu = 0.0
    valVTau = 0.0

    mN_grid = np.arange(0.1, 1.0, 0.05, dtype=np.float64)
    Ve_grid = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, np.sqrt(1e-5), 1e-2, np.sqrt(5.62e-4), np.sqrt(3.16e-4), np.sqrt(1.778e-4), np.sqrt(1e-3), np.sqrt(5.62e-3), np.sqrt(3.16e-3), np.sqrt(1.778e-3),  1e-1], dtype=np.float64)

    rows = compute_rows(mN_grid, Ve_grid, p_arr, theta_arr, NM_arr, valVMu, valVTau)
    out = pd.DataFrame(rows, columns=["mN (GeV)", "Ve^2", "Total Width", "BR Visible", "ctau (m)", "N_events"])
    out.to_csv("D-0-FCC-LLP1_gammaN_selected_numba_parallel.csv", index=False)