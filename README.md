# Holographic Entanglement of Fractal Regions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23189575.svg)](https://doi.org/10.5281/zenodo.23189575)

Code, data, figures and manuscript for

> **R. Chen**, *Holographic Entanglement of Fractal Regions: Complex Dimensions, Non-local Counterterms
> and Cosmic-Brane Multifractality* (2026). DOI: [10.5281/zenodo.23189575](https://doi.org/10.5281/zenodo.23189575)

## Summary

The paper studies holographic (Ryu–Takayanagi) entanglement entropy of boundary regions whose entangling
surface is a self-similar fractal, in Poincaré AdS<sub>d+1</sub>. Main results:

- **Holographic box counting.** The exact identity −ε∂<sub>ε</sub>S<sub>A</sub> = (R/4G<sub>N</sub>) Vol(γ<sub>A</sub> ∩ {z = ε}),
  an explicit positive radial smoothing kernel K<sub>z</sub>(y) ∝ z<sup>d</sup>/(|y|²+z²)<sup>d−1</sup>, and
  −ε∂<sub>ε</sub>S<sub>A</sub> ≍ N<sub>ε</sub>(∂A) for Weierstrass-type boundaries.
- **DSI locking.** For self-similar boundaries the complex dimensions governing S<sub>A</sub> are the zeros
  of the Moran function; bulk dynamics enters only through the residues (adapting the renewal-theoretic
  lattice/non-lattice dichotomy to the RT cross-section content).
- **Non-local counterterm** I<sub>ct</sub> = −(R/4G<sub>N</sub>)(−z<sub>c</sub>∂<sub>z<sub>c</sub></sub>)<sup>−1</sup>Π<sub>+</sub>Vol, and the precise reason local counterterms fail.
- **Marginal complex dimensions** (Re s = 0): a criterion for bounded log-periodic terms and an explicit
  lattice example, x⁵ + x − 1 = (x² − x + 1)(x³ + x² − 1).
- **Cosmic branes and multifractality**: refined Rényi entropy = singularity spectrum, S̃<sub>q</sub> = f(α<sub>q</sub>) ln(1/ε),
  with the d = 2 Calabrese–Lefevre spectrum as a consistency check.
- **Experimental fingerprint**: half-integer log-harmonics in lattice fractal-string resonators.

The manuscript states explicitly which results are proven, which rest on stated assumptions (notably the
near-boundary decoupling lemma) and which build on prior work.

## Repository structure

| Path | Contents |
|---|---|
| `paper/` | LaTeX source and compiled PDF of the manuscript |
| `code/holographic_fractal_rt.py` | Single script that produces every numerical result and figure |
| `figures/` | All figures, as vector PDF (used by the paper) and PNG |
| `data/` | Numerical data behind the figures and tables, as CSV (metadata in `#` header lines) |
| `results/` | Console output of each script mode (the numbers quoted in the paper) |

## Reproducing the results

Requirements: Python ≥ 3.10 and the packages in `requirements.txt`
(tested with Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1, SymPy 1.13.1, Matplotlib 3.8.4, mpmath 1.3.0).

```bash
pip install -r requirements.txt
python code/holographic_fractal_rt.py all        # or one of the modes below
```

Run from the repository root; figures are written to `figures/` and data to `data/`
(override with the environment variables `HFE_FIG_DIR`, `HFE_DATA_DIR`). Add `--show` to open the figures
interactively.

| Mode | Paper section | Figures | Data |
|---|---|---|---|
| `rt` | §3, Table 1 (also SymPy curvature checks) | `rt_profiles`, `rt_scaling` | `table1_exponents`, `content_vs_z`, `regulated_area_vs_z` |
| `ren` | §4.5 | `rt_renormalization` | `renormalization_curves`, `renormalization_complex_dims` |
| `renyi` | §6 | `renyi_multifractal` | `binomial_cantor_*`, `hyperbolic_branes` |
| `stage3` | §§5–7 | `dsi_complex_dims`, `multifractal_lockin` | `moran_poles_*`, `marginal_residual_*`, `multifractal_lockin_*` |
| `stage4` | §§6, 8 | `bcft_closure`, `resonator_fingerprint` | `bcft_cardy_spectrum`, `ising_cascade_*`, `resonator_heat_trace` |

The computations are deterministic; the logs in `results/` were produced with the versions listed above.

To rebuild the paper (pdfLaTeX, two passes):

```bash
cd paper
pdflatex Chen_2026_Holographic_Entanglement_of_Fractal_Regions.tex
pdflatex Chen_2026_Holographic_Entanglement_of_Fractal_Regions.tex
```

## Citation

If you use this code or data, please cite the paper (see also `CITATION.cff`):

```bibtex
@misc{Chen2026HolographicFractal,
  author = {Chen, Ruqing},
  title  = {Holographic Entanglement of Fractal Regions: Complex Dimensions,
            Non-local Counterterms and Cosmic-Brane Multifractality},
  year   = {2026},
  doi    = {10.5281/zenodo.23189575},
  url    = {https://doi.org/10.5281/zenodo.23189575}
}
```

## Contact

Ruqing Chen — GUT Geoservice Inc., Montreal — ruqing@hotmail.com
