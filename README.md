# Mid-Infrared Supercontinuum Simulation in an As₂S₃ Photonic Crystal Fiber

Numerical code for a GNLSE study of supercontinuum generation in a
dispersion-engineered As₂S₃ photonic crystal fiber, evaluated against the
carbon monoxide (4.67 µm) and RDX (5.88 µm) mid-infrared marker bands.

Companion code for the OMLET 2026 paper *Simulation of a Dispersion-Engineered
As₂S₃ Hollow-Defect Fiber Supercontinuum for Broadband Mid-Infrared
Chemical-Marker Screening*.

## Attribution of the fiber design

This is a **reproduction study**. The fiber geometry, dispersion, nonlinearity
and pump condition (d = 4.346 µm, Λ = 6.585 µm, ZDW = 2.96 µm,
D = 0.02567 ps nm⁻¹ km⁻¹, γ = 119.27 W⁻¹ km⁻¹, 3.30 µm / 1.5 kW) are **not
original to this work**. They are taken from:

> E. M. Sourav, M. N. Shakib, M. A. Islam, "Simulation of Broadband SC Source
> for Early Detection of Explosives and Hazardous Gases," Poster P0010,
> Department of Electrical and Electronic Engineering, Ahsanullah University of
> Science and Technology, Dhaka, Bangladesh, 2026.

What is original here is the solver, the convergence and sensitivity analysis,
the coherence study, and the marker-coverage evaluation.

## What this solves

The generalized nonlinear Schrödinger equation

```
∂A/∂z + (α/2)A + Σ_{k≥2} (i^{k-1} β_k / k!) ∂^k A/∂T^k
    = iγ (1 + (i/ω₀) ∂/∂T) A ∫ R(T') |A(T-T')|² dT'
```

is integrated with a fourth-order Runge–Kutta in the Interaction Picture
(RK4IP) scheme, including Kerr nonlinearity, self-steepening, a delayed Raman
response, a wavelength-dependent As₂S₃ loss edge, and one-photon-per-mode
quantum noise.

## Operating point

| Parameter | Value |
|---|---|
| Pump wavelength / peak power | 3.30 µm / 1.5 kW |
| Pump duration (assumed) | 50 fs FWHM, sech |
| Fiber length | 0.13 m |
| Zero-dispersion wavelength | 2.96 µm |
| Dispersion at pump | 0.02567 ps nm⁻¹ km⁻¹ |
| β₂ / β₃ / β₄ | −0.148 ps²/km, 2.263 ps³/km, −3.0 ps⁴/km |
| Nonlinear coefficient γ | 119.27 W⁻¹ km⁻¹ |
| Grid | 2¹⁴ points, 2 fs resolution, 6500 z-steps |

The pump duration is **not** a reported design value; it is an explicit
assumption, and `pcf_sensitivity.py` quantifies how strongly the result depends
on it.

## Scripts

| Script | Purpose |
|---|---|
| `src/pcf_sc_simulation.py` | Main RK4IP GNLSE solve. Writes the evolution map, terminal spectrum, dispersion/γ curves and marker traces to `simulation_data/`. |
| `src/pcf_sensitivity.py` | Step-size and grid convergence; sensitivity of the −20 dB span to pump duration, background loss and the multiphonon edge. |
| `src/pcf_coherence.py` | Shot-to-shot first-order coherence \|g₁₂(λ)\| over an ensemble of independent quantum-noise seeds. |

## Usage

```bash
pip install numpy matplotlib
cd src
python pcf_sc_simulation.py     # main solve  (~2 min)
python pcf_sensitivity.py       # convergence + sensitivity  (~15 min)
python pcf_coherence.py         # coherence ensemble  (~20 min)
```

Outputs are written to `src/simulation_data/`.

## Reference results

The baseline solve reproduces:

- **SC span** 1.00–6.38 µm at −20 dB (1.00–7.34 µm at −30 dB)
- **Soliton number** N ≈ 31, nonlinear length ≈ 5.6 mm
- **Marker power density** (dB below spectral peak):

  | Pump FWHM | Red edge | CO 4.67 µm | RDX 5.88 µm |
  |---|---|---|---|
  | 40 fs | 7.05 µm | −2.1 dB | −8.7 dB |
  | 50 fs | 6.38 µm | −0.9 dB | −13.9 dB |
  | 60 fs | 5.96 µm | −0.3 dB | −18.3 dB |

- **Coherence** power-weighted \|g₁₂\| = 0.9999 over 1.4–6.4 µm

## Scope and limitations

This is a simulation study. It establishes spectral compatibility between the
source and the selected marker bands; it is **not** a detection experiment.

- No fabricated fiber, gas cell, RDX sample or detector is involved. No
  detection limit, sensitivity, selectivity or false-alarm rate is reported.
- The modal problem is **not** solved here. Dispersion and nonlinearity are
  reconstructed from scalar design values (ZDW, D, γ); A_eff is inferred by
  inverting γ = 2πn₂/(λA_eff) rather than integrated from a computed field.
- β(ω) is a fourth-order expansion applied across 1.0–7.3 µm. Higher-order
  truncation is a known source of non-physical bandwidth in GNLSE studies.
- γ carries a 1/λ dependence via the self-steepening factor. The physical
  γ falls faster because A_eff grows with wavelength; under γ ∝ λ^-1.5 the
  −20 dB edge moves to 5.26 µm and the RDX band drops to −34 dB. **The RDX
  coverage result is model-dependent**; the CO result is robust across every
  variant tested.
- The marker traces are parameterized band-shaped notches imposed on the solved
  spectrum, **not** a line-by-line Beer–Lambert evaluation against HITRAN or an
  FTIR library. Concentration and path length are not specified, so notch
  depths indicate band placement, not absorbance.
- Atmospheric interferents (CO₂ near 4.3 µm, H₂O) are not modelled and would
  dominate an uncorrected measurement at the CO wavelength.

## References

- J. Hult, "A fourth-order Runge–Kutta in the interaction picture method for
  simulating supercontinuum generation in optical fibers," *J. Lightwave
  Technol.* **25**(12), 3770–3775, 2007.
- J. M. Dudley, G. Genty, S. Coen, "Supercontinuum generation in photonic
  crystal fiber," *Rev. Mod. Phys.* **78**(4), 1135–1184, 2006.

## Requirements

Python 3.9+, NumPy, Matplotlib (used only by `pcf_coherence.py`).
