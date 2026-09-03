"""
Numerical-validation study for the PCF supercontinuum solver:
  (a) step-size and grid convergence of the RK4IP GNLSE solve, and
  (b) sensitivity of the -20 dB span to the three assumed inputs
      (pump duration T0, background loss, multiphonon edge).

Reuses the same physics as pcf_sc_simulation.py, wrapped in solve(...).
Writes simulation_data/convergence.csv and sensitivity.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "simulation_data"; DATA.mkdir(exist_ok=True)

c = 299_792_458.0
lam0 = 3.30e-6; w0 = 2 * np.pi * c / lam0
P0 = 1500.0; gamma0 = 119.27e-3; L = 0.13
lam_zdw = 2.96e-6; D_pump = 0.02567
fR, tau1, tau2 = 0.11, 15.5e-15, 230.5e-15


def solve(T0=28.4e-15, alpha_bg=1.0, ir_edge=6.85, N=2**14, dz_steps=6500, seed=20260215):
    """Return (lam_um sorted, terminal dB spectrum) from an RK4IP GNLSE solve."""
    D_si = D_pump * 1e-6
    beta2 = -D_si * lam0**2 / (2 * np.pi * c)
    w_zdw = 2 * np.pi * c / lam_zdw
    beta3 = -beta2 / (w_zdw - w0); beta4 = -3.0e-57
    dt = 2.0e-15
    T = (np.arange(N) - N / 2) * dt
    w = 2 * np.pi * np.fft.fftfreq(N, dt)
    nu_all = (w + w0) / (2 * np.pi)
    lam_all = np.where(nu_all > 1e9, c / np.maximum(nu_all, 1e9), 1e-3)
    lam_all_um = lam_all * 1e6
    alpha_dBm = (alpha_bg + np.exp(np.clip((lam_all_um - ir_edge) / 0.26, -50, 50))
                 + np.exp(np.clip((1.15 - lam_all_um) / 0.10, -50, 50)))
    alpha = np.clip(alpha_dBm / (10 * np.log10(np.e)), 0, 4000.0)
    Dop = -alpha / 2 + 1j * (beta2 / 2 * w**2 + beta3 / 6 * w**3 + beta4 / 24 * w**4)
    hR = (tau1**2 + tau2**2) / (tau1 * tau2**2) * np.exp(-T / tau2) * np.sin(T / tau1)
    hR[T < 0] = 0.0; hR = np.fft.ifftshift(hR); RW = np.fft.fft(hR) * dt
    ss = 1 + w / w0

    def NL(Aw):
        A = np.fft.ifft(Aw); P = np.abs(A) ** 2
        conv = np.fft.ifft(RW * np.fft.fft(P)) * dt
        return 1j * gamma0 * ss * np.fft.fft(A * ((1 - fR) * P + fR * conv))

    def step(Aw, dz):
        expH = np.exp(Dop * dz / 2); AwI = expH * Aw
        k1 = expH * (dz * NL(Aw)); k2 = dz * NL(AwI + k1 / 2)
        k3 = dz * NL(AwI + k2 / 2); k4 = dz * NL(expH * (AwI + k3))
        return expH * (AwI + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6

    rng = np.random.default_rng(seed)
    A0 = np.sqrt(P0) / np.cosh(T / T0)
    hbar = 1.054_571_817e-34; dw = 2 * np.pi / (N * dt)
    photon = np.sqrt(hbar * np.abs(w + w0) / (2 * dw))
    Aw = np.fft.fft(A0) + photon * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    dz = L / dz_steps
    for _ in range(dz_steps):
        Aw = step(Aw, dz)
    wsh = np.fft.fftshift(w); nu = (wsh + w0) / (2 * np.pi); valid = nu > 0
    lam = c / nu[valid]; order = np.argsort(lam); lam_um = lam[order] * 1e6
    S = (np.abs(np.fft.fftshift(Aw)) ** 2)[valid][order] * (nu[valid][order] ** 2)
    disp = (lam_um >= 1.0) & (lam_um <= 8.0)
    x = lam_um[disp]; y = S[disp] / S[disp].max()
    ydb = 10 * np.log10(np.clip(y, 1e-7, None)); ydb -= ydb.max()
    return x, ydb


def span20(x, ydb):
    a = np.where(ydb >= -20)[0]
    return (x[a[0]], x[a[-1]]) if a.size else (np.nan, np.nan)


def main():
    # baseline
    xb, yb = solve()
    b = span20(xb, yb)
    print(f"baseline span@-20dB: {b[0]:.2f}-{b[1]:.2f} um")

    # (a) convergence: finer step and finer grid, compare span + spectral RMS
    conv = [("baseline (N=2^14, 6500 steps)", *b, 0.0)]
    for label, kw in [("half step (13000)", dict(dz_steps=13000)),
                      ("quarter step (3250)", dict(dz_steps=3250)),
                      ("fine grid (N=2^15)", dict(N=2**15))]:
        x, y = solve(**kw)
        s = span20(x, y)
        yi = np.interp(xb, x, y)
        rms = float(np.sqrt(np.mean((yi - yb) ** 2)))
        conv.append((label, s[0], s[1], rms))
        print(f"{label}: span {s[0]:.2f}-{s[1]:.2f} um, spectral RMS vs baseline {rms:.2f} dB")
    import csv
    with open(DATA / "convergence.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "blue_um", "red_um", "rms_db"]); w.writerows(conv)

    # (b) sensitivity to the three assumptions
    sens = [("baseline", 50, 1.0, 6.85, *b)]
    for label, kw, note in [
        ("T0 = 40 fs", dict(T0=22.7e-15), (40, 1.0, 6.85)),
        ("T0 = 60 fs", dict(T0=34.0e-15), (60, 1.0, 6.85)),
        ("loss x2 (2 dB/m)", dict(alpha_bg=2.0), (50, 2.0, 6.85)),
        ("IR edge 6.6 um", dict(ir_edge=6.6), (50, 1.0, 6.6)),
        ("IR edge 7.1 um", dict(ir_edge=7.1), (50, 1.0, 7.1)),
    ]:
        x, y = solve(**kw); s = span20(x, y)
        sens.append((label, *note, s[0], s[1]))
        print(f"{label}: span {s[0]:.2f}-{s[1]:.2f} um")
    with open(DATA / "sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "T0_fs", "loss_dBm", "ir_edge_um", "blue_um", "red_um"])
        w.writerows(sens)
    print("saved convergence.csv and sensitivity.csv")


if __name__ == "__main__":
    main()
