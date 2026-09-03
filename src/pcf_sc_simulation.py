"""
Physical supercontinuum simulation for a dispersion-engineered As2S3 PCF.

This is a genuine numerical solve, not a cosmetic reconstruction:

  * Dispersion beta(omega) is built from the reported operating point
    (D = 0.02567 ps/nm/km at the 3.30 um pump) and the reported
    zero-dispersion wavelength (ZDW = 2.96 um), giving beta2, beta3, beta4.
  * The generalized nonlinear Schrodinger equation (GNLSE) is integrated with
    a 4th-order Runge-Kutta in the Interaction Picture (RK4IP, Hult 2007),
    including Kerr nonlinearity, self-steepening (shock term), and a delayed
    Raman response, seeded with one-photon-per-mode quantum noise.
  * The supercontinuum evolution map and terminal spectrum are extracted from
    the SAME propagated field; every CSV written here is a product of this
    solve.

  NOTE ON THE MARKER STAGE: the CO / RDX traces are NOT a line-by-line
  Beer-Lambert evaluation against HITRAN or an FTIR library.  They are
  parameterized band-shaped notches (see marker() below) imposed on the solved
  terminal spectrum at the marker centres, illustrating band placement only.
  Concentration c_a and path length L_a are not specified, so the notch depths
  are NOT predicted absorbances.

The one operating input the design does not fix is the pump duration; it is
declared here as an explicit assumption (T0).  See pcf_sensitivity.py for how
strongly the result depends on it.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "simulation_data"
DATA.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Physical constants and the reported operating point for this design
# ----------------------------------------------------------------------------
c = 299_792_458.0                     # m/s
lam0 = 3.30e-6                         # pump wavelength (m)
w0 = 2 * np.pi * c / lam0             # pump angular frequency (rad/s)
P0 = 1500.0                            # pump peak power (W)  [1.5 kW]
gamma0 = 119.27e-3                     # nonlinear coefficient (W^-1 m^-1) [119.27 W^-1 km^-1]
L = 0.13                               # propagation length (m)
lam_zdw = 2.96e-6                      # zero-dispersion wavelength (m)
D_pump = 0.02567                       # dispersion at pump  ps/(nm km)

# Pump duration is NOT a reported design value -> declared assumption.
T0 = 28.4e-15                          # sech half-width (s); FWHM ~ 1.763*T0 ~ 50 fs

# As2S3 approximate Raman model and background loss.
fR = 0.11
tau1, tau2 = 15.5e-15, 230.5e-15
alpha_dB = 1.0                         # dB/m background loss (MIR As2S3, conservative)
alpha = alpha_dB / (10 * np.log10(np.e))  # 1/m

# ----------------------------------------------------------------------------
# Dispersion: beta2 from reported D at pump, beta3 from the ZDW constraint,
# beta4 a small higher-order term giving a realistic long-wave rolloff.
# ----------------------------------------------------------------------------
# D [s/m^2] from ps/(nm km):  1 ps/(nm km) = 1e-6 s/m^2
D_si = D_pump * 1e-6
beta2 = -D_si * lam0**2 / (2 * np.pi * c)          # s^2/m  (anomalous, <0)
# ZDW: beta2(w_zdw)=0 with linear model beta2(w)=beta2+beta3*(w-w0)
w_zdw = 2 * np.pi * c / lam_zdw
beta3 = -beta2 / (w_zdw - w0)                       # s^3/m  (>0)
beta4 = -3.0e-57                                    # s^4/m  (keeps a single ZDW in-band)
betas = [beta2, beta3, beta4]

print(f"beta2 = {beta2:.4e} s^2/m ({beta2*1e27:.4f} ps^2/km)")
print(f"beta3 = {beta3:.4e} s^3/m ({beta3*1e42:.4f} ps^3/km)")
print(f"L_NL = {1/(gamma0*P0)*1e3:.3f} mm | L_D = {T0**2/abs(beta2):.4f} m | "
      f"N_sol = {np.sqrt(gamma0*P0*T0**2/abs(beta2)):.1f}")

# ----------------------------------------------------------------------------
# Time / frequency grid
# ----------------------------------------------------------------------------
N = 2**14
dt = 2.0e-15                          # s
T = (np.arange(N) - N / 2) * dt       # time (s)
w = 2 * np.pi * np.fft.fftfreq(N, dt) # angular frequency offset (rad/s)
w_abs = w + w0                        # absolute angular frequency

# Wavelength-dependent loss alpha(lambda): background + As2S3 multiphonon IR edge
# (rises steeply beyond ~6.7 um) + short-wave edge; also acts as an absorbing
# boundary that suppresses spectral wrap-around at the grid edges.
nu_all = (w + w0) / (2 * np.pi)
lam_all = np.where(nu_all > 1e9, c / np.maximum(nu_all, 1e9), 1e-3)   # m
lam_all_um = lam_all * 1e6
alpha_dBm = (
    alpha_dB
    + np.exp(np.clip((lam_all_um - 6.85) / 0.26, -50, 50))   # multiphonon IR edge (As2S3)
    + np.exp(np.clip((1.15 - lam_all_um) / 0.10, -50, 50))   # short-wave edge
)
alpha_arr = alpha_dBm / (10 * np.log10(np.e))  # 1/m
alpha_arr = np.clip(alpha_arr, 0, 4000.0)      # cap for numerical stability

# Linear operator (dispersion + wavelength-dependent loss), frequency domain
Dop = -alpha_arr / 2 + 1j * (
    betas[0] / 2 * w**2 + betas[1] / 6 * w**3 + betas[2] / 24 * w**4
)

# Raman response function h_R(t) (single damped oscillator), -> frequency
hR = (tau1**2 + tau2**2) / (tau1 * tau2**2) * np.exp(-T / tau2) * np.sin(T / tau1)
hR[T < 0] = 0.0
hR = np.fft.ifftshift(hR)
RW = np.fft.fft(hR) * dt              # normalized Raman spectrum

ss = 1 + w / w0                       # self-steepening (shock) factor


def nonlinear(Aw: np.ndarray) -> np.ndarray:
    """Nonlinear operator in the frequency domain for a given spectral field."""
    A = np.fft.ifft(Aw)
    P = np.abs(A) ** 2
    conv = np.fft.ifft(RW * np.fft.fft(P)) * dt
    M = A * ((1 - fR) * P + fR * conv)
    return 1j * gamma0 * ss * np.fft.fft(M)


def rk4ip(Aw: np.ndarray, dz: float) -> np.ndarray:
    """One RK4 Interaction-Picture step (Hult 2007)."""
    expH = np.exp(Dop * dz / 2)
    AwI = expH * Aw
    k1 = expH * (dz * nonlinear(Aw))
    k2 = dz * nonlinear(AwI + k1 / 2)
    k3 = dz * nonlinear(AwI + k2 / 2)
    k4 = dz * nonlinear(expH * (AwI + k3))
    return expH * (AwI + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6


# ----------------------------------------------------------------------------
# Input field: sech pulse + one-photon-per-mode quantum noise
# ----------------------------------------------------------------------------
rng = np.random.default_rng(20260215)
A0 = np.sqrt(P0) / np.cosh(T / T0)
# one photon per mode spectral noise
hbar = 1.054_571_817e-34
dw = 2 * np.pi / (N * dt)
photon = np.sqrt(hbar * np.abs(w_abs) / (2 * dw))
noise = photon * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
Aw = np.fft.fft(A0) + noise
A = np.fft.ifft(Aw)

# ----------------------------------------------------------------------------
# Propagate, storing evolution snapshots
# ----------------------------------------------------------------------------
nz = 6500
dz = L / nz
nsave = 300
save_every = nz // nsave
zsave = [0.0]
spec_save = [np.fft.fftshift(Aw).copy()]
Aw = np.fft.fft(A)
for i in range(1, nz + 1):
    Aw = rk4ip(Aw, dz)
    if i % save_every == 0:
        zsave.append(i * dz)
        spec_save.append(np.fft.fftshift(Aw).copy())

zsave = np.array(zsave)
spec = np.array(spec_save)               # (nsave+1, N) complex, fftshifted

# ----------------------------------------------------------------------------
# Convert spectral field to wavelength-domain intensity (dB)
# ----------------------------------------------------------------------------
wsh = np.fft.fftshift(w)
nu_abs = (wsh + w0) / (2 * np.pi)        # absolute frequency (Hz)
valid = nu_abs > 0
lam = c / nu_abs[valid]                   # wavelength (m)
order = np.argsort(lam)
lam_um = lam[order] * 1e6

# spectral energy density in wavelength: |A(w)|^2 * (dw/dlam) ~ |A|^2 * w^2
Sw = np.abs(spec) ** 2
Slam = Sw[:, valid][:, order] * (nu_abs[valid][order] ** 2)   # Jacobian nu^2 ~ 1/lam^2 scaling

def to_db(x, floor=1e-8):
    x = x / np.max(x)
    return 10 * np.log10(np.clip(x, floor, None))

# restrict to a display window 1.0 - 8.0 um
disp = (lam_um >= 1.0) & (lam_um <= 8.0)
lam_disp = lam_um[disp]
map_db = np.array([to_db(row[disp] + 1e-30, floor=1e-7) for row in Slam])
map_db -= map_db.max()
terminal = map_db[-1].copy()
terminal -= terminal.max()

# ----------------------------------------------------------------------------
# Measured supercontinuum span (real solver output) at -20 / -30 dB
# ----------------------------------------------------------------------------
def span_at(level):
    above = np.where(terminal >= level)[0]
    if above.size == 0:
        return (np.nan, np.nan)
    return (lam_disp[above[0]], lam_disp[above[-1]])

span20 = span_at(-20)
span30 = span_at(-30)
print(f"SC span @ -20 dB: {span20[0]:.2f}-{span20[1]:.2f} um")
print(f"SC span @ -30 dB: {span30[0]:.2f}-{span30[1]:.2f} um")

# ----------------------------------------------------------------------------
# Save simulation data (real arrays)
# ----------------------------------------------------------------------------
np.savetxt(DATA / "sc_output_generated.csv",
           np.column_stack([lam_disp * 1000, terminal]),
           delimiter=",", header="wavelength_nm,relative_power_db", comments="")

# thinned evolution grid for CSV (keep file reasonable)
li = np.linspace(0, lam_disp.size - 1, 400).astype(int)
zi = np.arange(zsave.size)
rows = []
for zz in zi:
    for xx in li:
        rows.append((zsave[zz], lam_disp[xx] * 1000, map_db[zz, xx]))
np.savetxt(DATA / "sc_evolution_generated.csv", np.array(rows),
           delimiter=",", header="distance_m,wavelength_nm,relative_power_db", comments="")

# dispersion + gamma curves (real model)
lam_axis = np.linspace(1.4e-6, 6.6e-6, 700)
w_axis = 2 * np.pi * c / lam_axis - w0
beta2_axis = betas[0] + betas[1] * w_axis + betas[2] / 2 * w_axis**2
D_axis = -2 * np.pi * c / lam_axis**2 * beta2_axis            # s/m^2
D_axis_ps = D_axis * 1e6                                       # ps/(nm km)
# Plotted gamma(lambda).  NOTE: in the SOLVER the 1/lam dependence enters via the
# self-steepening factor ss = 1 + w/w0, which is algebraically identical to
# gamma ~ 1/lam.  A physical Aeff(lam) growth would make gamma fall faster than
# 1/lam.  See the README for the magnitude of that sensitivity.
gamma_axis = gamma0 * (lam0 / lam_axis)
np.savetxt(DATA / "dispersion_generated.csv",
           np.column_stack([lam_axis * 1e6, D_axis_ps]),
           delimiter=",", header="wavelength_um,dispersion_ps_nm_km", comments="")
np.savetxt(DATA / "nonlinearity_generated.csv",
           np.column_stack([lam_axis * 1e6, gamma_axis * 1e3]),
           delimiter=",", header="wavelength_um,gamma_W_km", comments="")

# ----------------------------------------------------------------------------
# Marker band placement on the REAL terminal spectrum.
# NOTE: parameterized notch, NOT a computed Beer-Lambert absorbance (see header).
# ----------------------------------------------------------------------------
def marker(depth_db, center_um, width_um):
    att = depth_db * np.exp(-0.5 * ((lam_disp - center_um) / width_um) ** 2)
    return terminal - att

co = marker(22.0, 4.67, 0.045)
rdx = marker(24.0, 5.88, 0.060)
np.savetxt(DATA / "co_absorption_generated.csv",
           np.column_stack([lam_disp * 1000, terminal, co]),
           delimiter=",", header="wavelength_nm,source_db,transmitted_db", comments="")
np.savetxt(DATA / "rdx_absorption_generated.csv",
           np.column_stack([lam_disp * 1000, terminal, rdx]),
           delimiter=",", header="wavelength_nm,source_db,transmitted_db", comments="")

# expose arrays for downstream analysis
np.savez(DATA / "_sim_arrays.npz",
         lam_disp=lam_disp, zsave=zsave, map_db=map_db, terminal=terminal,
         lam_axis=lam_axis * 1e6, D_axis=D_axis_ps, gamma_axis=gamma_axis * 1e3,
         co=co, rdx=rdx, span20=np.array(span20), span30=np.array(span30),
         zdw=lam_zdw * 1e6, gamma_pt=gamma0 * 1e3, lam0=lam0 * 1e6)
print("Saved simulation arrays and CSVs to", DATA)
