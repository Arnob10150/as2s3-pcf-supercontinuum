"""
Shot-to-shot coherence of the simulated supercontinuum.

Runs the RK4IP GNLSE solve for M independent one-photon-per-mode noise seeds and
computes the modulus of the complex degree of first-order coherence,
    |g12(lambda)| = | <E_i^*(lambda) E_j(lambda)>_{i!=j} | / <|E(lambda)|^2>,
using the efficient estimator num = (|sum E|^2 - sum|E|^2)/(M(M-1)).
Reports the power-weighted mean coherence over the useful band and writes a
plot of coherence vs wavelength plus coherence.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "simulation_data"; FIG = ROOT / "figures"
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

c = 299_792_458.0
lam0 = 3.30e-6; w0 = 2 * np.pi * c / lam0
P0 = 1500.0; gamma0 = 119.27e-3; L = 0.13
lam_zdw = 2.96e-6; D_pump = 0.02567
fR, tau1, tau2 = 0.11, 15.5e-15, 230.5e-15
T0 = 28.4e-15                      # 50 fs FWHM, same as main solve
N = 2**14; dt = 2.0e-15; NZ = 6500


def solve_field(seed):
    """Return (lam_um sorted, complex spectral field E(lambda) on display grid)."""
    D_si = D_pump * 1e-6
    beta2 = -D_si * lam0**2 / (2 * np.pi * c)
    beta3 = -beta2 / (2 * np.pi * c / lam_zdw - w0); beta4 = -3.0e-57
    T = (np.arange(N) - N / 2) * dt
    w = 2 * np.pi * np.fft.fftfreq(N, dt)
    nu_all = (w + w0) / (2 * np.pi)
    lam_all_um = np.where(nu_all > 1e9, c / np.maximum(nu_all, 1e9), 1e-3) * 1e6
    alpha_dBm = (1.0 + np.exp(np.clip((lam_all_um - 6.85) / 0.26, -50, 50))
                 + np.exp(np.clip((1.15 - lam_all_um) / 0.10, -50, 50)))
    alpha = np.clip(alpha_dBm / (10 * np.log10(np.e)), 0, 4000.0)
    Dop = -alpha / 2 + 1j * (beta2 / 2 * w**2 + beta3 / 6 * w**3 + beta4 / 24 * w**4)
    hR = (tau1**2 + tau2**2) / (tau1 * tau2**2) * np.exp(-T / tau2) * np.sin(T / tau1)
    hR[T < 0] = 0.0; hR = np.fft.ifftshift(hR); RW = np.fft.fft(hR) * dt
    ss = 1 + w / w0

    def NL(Aw):
        A = np.fft.ifft(Aw); Pp = np.abs(A) ** 2
        conv = np.fft.ifft(RW * np.fft.fft(Pp)) * dt
        return 1j * gamma0 * ss * np.fft.fft(A * ((1 - fR) * Pp + fR * conv))

    def step(Aw, dz):
        eH = np.exp(Dop * dz / 2); AwI = eH * Aw
        k1 = eH * (dz * NL(Aw)); k2 = dz * NL(AwI + k1 / 2)
        k3 = dz * NL(AwI + k2 / 2); k4 = dz * NL(eH * (AwI + k3))
        return eH * (AwI + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6

    rng = np.random.default_rng(seed)
    A0 = np.sqrt(P0) / np.cosh(T / T0)
    hbar = 1.054_571_817e-34
    # one photon per mode (random phase) in the numpy-fft convention Aw=fft(A):
    # physical spectral field is A_tilde = fft(A)*dt, so vacuum amplitude in this
    # convention is sqrt(hbar*omega*N/dt); split over real/imag with /sqrt(2).
    photon = np.sqrt(hbar * np.abs(w + w0) * N / dt)
    Aw = np.fft.fft(A0) + photon * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    dz = L / NZ
    for _ in range(NZ):
        Aw = step(Aw, dz)
    wsh = np.fft.fftshift(w); nu = (wsh + w0) / (2 * np.pi); valid = nu > 0
    lam = c / nu[valid]; order = np.argsort(lam)
    lam_um = lam[order] * 1e6
    E = np.fft.fftshift(Aw)[valid][order]
    disp = (lam_um >= 1.0) & (lam_um <= 8.0)
    return lam_um[disp], E[disp]


def main(M=20):
    fields = []
    lam = None
    for s in range(M):
        lam, E = solve_field(seed=90000 + s)
        fields.append(E)
        print(f"seed {s+1}/{M} done")
    F = np.array(fields)                      # (M, Nlam) complex
    S = np.abs(F) ** 2
    den = S.mean(0)
    num = (np.abs(F.sum(0)) ** 2 - S.sum(0)) / (M * (M - 1))
    g12 = np.abs(num) / np.clip(den, 1e-30, None)
    g12 = np.clip(g12, 0, 1)

    meanS = den / den.max()
    band = (lam >= 1.4) & (lam <= 6.4)
    wmean = float(np.sum(g12[band] * meanS[band]) / np.sum(meanS[band]))
    # coherence over the pump-side (1.4-4) vs red edge (4-6.4)
    b1 = (lam >= 1.4) & (lam < 4.0); b2 = (lam >= 4.0) & (lam <= 6.4)
    c1 = float(np.sum(g12[b1] * meanS[b1]) / np.sum(meanS[b1]))
    c2 = float(np.sum(g12[b2] * meanS[b2]) / np.sum(meanS[b2]))
    print(f"\npower-weighted mean |g12| over 1.4-6.4 um = {wmean:.3f}")
    print(f"  1.4-4.0 um: {c1:.3f}   4.0-6.4 um: {c2:.3f}")

    np.savetxt(DATA / "coherence.csv",
               np.column_stack([lam, g12, meanS]),
               delimiter=",", header="wavelength_um,g12,mean_norm_power", comments="")

    # figure
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "savefig.dpi": 360})
    fig, ax1 = plt.subplots(figsize=(3.35, 2.3))
    md = 10 * np.log10(np.clip(meanS, 1e-6, None))
    ax1.plot(lam, md, color="#1f5aa6", lw=0.9, label="mean spectrum")
    ax1.set_xlabel("Wavelength ($\\mu$m)"); ax1.set_ylabel("Power (dB)", color="#1f5aa6")
    ax1.set_xlim(1, 7); ax1.set_ylim(-60, 3); ax1.tick_params(axis="y", labelcolor="#1f5aa6")
    ax2 = ax1.twinx()
    ax2.plot(lam, g12, color="#c0392b", lw=1.0, label="$|g_{12}|$")
    ax2.set_ylabel("$|g_{12}(\\lambda)|$", color="#c0392b"); ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    for m, cc in [(4.67, "#2e7d32"), (5.88, "#6a1b9a")]:
        ax1.axvline(m, color=cc, ls=":", lw=0.7)
    fig.tight_layout()
    fig.savefig(FIG / "coherence.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / "coherence.png", bbox_inches="tight", pad_inches=0.02)
    print("saved coherence plot and coherence.csv")


if __name__ == "__main__":
    main()
