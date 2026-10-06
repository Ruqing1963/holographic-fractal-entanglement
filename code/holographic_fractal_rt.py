"""
Holographic RT surfaces for fractal entangling boundaries: curvature checks, minimal-surface
profiles, divergence exponents, complex dimensions, Renyi/cosmic-brane spectra.

Companion code for: R. Chen, "Holographic Entanglement of Fractal Regions" (2026),
DOI: 10.5281/zenodo.23189575

Model
-----
AdS_{d+1} Poincare patch:  ds^2 = (R^2/z^2)(dz^2 - dt^2 + dx^2 + dy^2 + dw_i^2),  i = 1..d-3
Boundary region A = { x < f0(y) } (x flat directions w_i), f0 a Weierstrass-type fractal curve:
    f0(y) = A0 * sum_n a^n cos(b^n k0 y),   a = b^(D-2)   =>  graph box dimension D in (1,2)
Linearised RT equation for x = f(y,z):  f_zz - (d-1)/z f_z + f_yy = 0
Single-mode solution (normalised phi(0)=1):  φ_d(kz) = (kz)^{d/2} K_{d/2}(kz) / (2^{d/2-1} Γ(d/2))
    for d=3 it is elementary: φ_3 = (1+kz) e^{-kz}
=> only modes with k <~ 1/z survive at depth z: radial coordinate = resolution.

Note: for fractal boundaries slopes are not small at small scales, so the linearised surface is not
an exact minimal surface; it captures the scaling (mode-by-mode decay, content scaling).
The exact nonlinear surface requires relaxation / Surface Evolver.

Usage
-----
    python holographic_fractal_rt.py [rt|ren|renyi|stage3|stage4|all] [--show]

Figures are written to ./figures (PNG and PDF), numerical data to ./data (CSV).
The output directories can be changed with the environment variables HFE_FIG_DIR and HFE_DATA_DIR.
"""
import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.special import kv, gamma, gammaln
from scipy.optimize import brentq

FIG_DIR = os.environ.get("HFE_FIG_DIR", "figures")
DATA_DIR = os.environ.get("HFE_DATA_DIR", "data")
MODES = ("rt", "ren", "renyi", "stage3", "stage4", "all")


def savefig(fig, name):
    """Save a figure as PNG (150 dpi) and vector PDF."""
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=150)


def savedata(name, columns, cols, meta=()):
    """Write equal-length columns to DATA_DIR/name.csv; `meta` lines are written as '# ' comments."""
    os.makedirs(DATA_DIR, exist_ok=True)
    arr = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    with open(os.path.join(DATA_DIR, f"{name}.csv"), "w", encoding="utf-8", newline="\n") as fh:
        for line in meta:
            fh.write(f"# {line}\n")
        fh.write(",".join(columns) + "\n")
        np.savetxt(fh, arr, delimiter=",", fmt="%.10g")

# ---------------------------------------------------------------- 1. symbolic curvature
def verify_curvature():
    z, t, x1, x2, x3, R = sp.symbols('z t x1 x2 x3 R', positive=True)
    X = [z, t, x1, x2, x3]
    g = sp.diag(1, -1, 1, 1, 1) * R**2 / z**2
    gi = g.inv()
    n = 5
    Gam = [[[sp.simplify(sum(gi[a, e] * (sp.diff(g[e, b], X[c]) + sp.diff(g[e, c], X[b])
                                         - sp.diff(g[b, c], X[e])) for e in range(n)) / 2)
             for c in range(n)] for b in range(n)] for a in range(n)]
    print("Non-zero Christoffel symbols:")
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                if Gam[a][b][c] != 0:
                    print(f"  Γ^{X[a]}_{{{X[b]}{X[c]}}} = {Gam[a][b][c]}")

    def Riem(a, b, c, d_):  # R^a_{bcd}
        r = sp.diff(Gam[a][b][d_], X[c]) - sp.diff(Gam[a][b][c], X[d_])
        r += sum(Gam[a][c][e] * Gam[e][b][d_] - Gam[a][d_][e] * Gam[e][b][c] for e in range(n))
        return sp.simplify(r)

    ok = True
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d_ in range(n):
                    # maximally symmetric space: R^a_{bcd} = -(1/R^2)(δ^a_c g_bd - δ^a_d g_bc)
                    target = -(sp.KroneckerDelta(a, c) * g[b, d_] - sp.KroneckerDelta(a, d_) * g[b, c]) / R**2
                    if sp.simplify(Riem(a, b, c, d_) - target) != 0:
                        ok = False
    Ric = sp.Matrix(n, n, lambda b, d_: sp.simplify(sum(Riem(a, b, a, d_) for a in range(n))))
    Rs = sp.simplify(sum(gi[b, d_] * Ric[b, d_] for b in range(n) for d_ in range(n)))
    print("R^a_{bcd} = -(δ^a_c g_bd - δ^a_d g_bc)/R^2 :", ok)
    print("Ricci = -4/R^2 g :", sp.simplify(Ric + 4 * g / R**2) == sp.zeros(n, n))
    print("Ricci scalar =", Rs)

    # check of the d=3 single-mode solution of the linearised RT equation
    k = sp.symbols('k', positive=True)
    phi = (1 + k * z) * sp.exp(-k * z)
    print("phi_3 satisfies φ'' - (2/z)φ' - k²φ = 0 :",
          sp.simplify(sp.diff(phi, z, 2) - 2 / z * sp.diff(phi, z) - k**2 * phi) == 0)


# ---------------------------------------------------------------- 2. bulk profiles
def phi(x, d):
    nu = d / 2
    x = np.maximum(x, 1e-300)
    return x**nu * kv(nu, x) / (2**(nu - 1) * gamma(nu))


def dphi(x, d):  # d/dx [x^ν K_ν] = -x^ν K_{ν-1}
    nu = d / 2
    x = np.maximum(x, 1e-300)
    return -x**nu * kv(nu - 1, x) / (2**(nu - 1) * gamma(nu))


class FractalRT:
    def __init__(self, D, d=3, b=3, nmodes=6, A0=1.0, k0=1.0, ny=2**15):
        self.D, self.d = D, d
        self.a = b ** (D - 2.0)
        self.k = k0 * b ** np.arange(nmodes)
        self.amp = A0 * self.a ** np.arange(nmodes)
        self.y = np.linspace(0, 2 * np.pi / k0, ny, endpoint=False)
        self.dy = self.y[1] - self.y[0]

    def fields(self, z):
        f = np.zeros_like(self.y); fy = np.zeros_like(self.y); fz = np.zeros_like(self.y)
        for A, k in zip(self.amp, self.k):
            c, s = np.cos(k * self.y), np.sin(k * self.y)
            p = phi(k * z, self.d)
            f += A * p * c
            fy += -A * k * p * s
            fz += A * k * dphi(k * z, self.d) * c
        return f, fy, fz

    def area_density(self, z):
        """L(z) = ∫ dy sqrt(1 + f_y^2 + f_z^2): (coordinate) length of the cross-section at depth z"""
        _, fy, fz = self.fields(z)
        return np.sum(np.sqrt(1 + fy**2 + fz**2)) * self.dy


# ================================================================ stage 2
# ---------------------------------------------------------------- 3. complex dimensions / non-local counterterm / log-periodicity
def renormalization_demo(D=1.5, d=3, b=3, M=3, nmodes=7):
    """
    Renewal equation (AdS dilatation isometry + self-affinity):  L(z) ≈ b^{D-1} L(bz) + g(z)
    => Mellin poles (complex dimensions) s_m = D - i m p,  p = 2π/ln b
    => L(z) = c_s + z^{1-D} Σ_m C_m z^{i m p} + ...
    Non-local counterterm (subtract only divergent poles with Re s > 0):
        I_ct(z_c) = -R^2 [ c_s / z_c + Σ_m Re( C_m z_c^{-s_m} / s_m ) ]
    """
    m = FractalRT(D, d=d, b=b, nmodes=nmodes, ny=2**16)
    zs = np.logspace(np.log10(3 / m.k[-1]), np.log10(0.3), 200)
    L = np.array([m.area_density(z) for z in zs])
    lz = np.log(zs)
    p = 2 * np.pi / np.log(b)

    def design(Dx):
        cols = [np.ones_like(zs), zs**(1 - Dx)]
        for k in range(1, M + 1):
            cols += [zs**(1 - Dx) * np.cos(k * p * lz), zs**(1 - Dx) * np.sin(k * p * lz)]
        return np.stack(cols, 1)

    # scan the effective dimension (finite modes / non-homogeneity shift it from D)
    best = None
    for Dx in np.linspace(D - 0.2, D + 0.2, 401):
        X = design(Dx)
        coef, *_ = np.linalg.lstsq(X, L, rcond=None)
        res = np.linalg.norm(X @ coef - L) / np.linalg.norm(L)
        if best is None or res < best[0]:
            best = (res, Dx, coef)
    res, Dfit, coef = best
    c_s, c0 = coef[0], coef[1]
    C = [coef[2 + 2 * (k - 1)] - 1j * coef[3 + 2 * (k - 1)] for k in range(1, M + 1)]
    s = [Dfit - 1j * k * p for k in range(1, M + 1)]

    def I_ct(zc):
        out = c_s / zc + c0 * zc**(-Dfit) / Dfit
        for Ck, sk in zip(C, s):
            out = out + np.real(Ck * zc**(-sk) / sk)
        return out

    # regulated area Area(z_c) = ∫_{z_c}^{Z} z^{-2} L dz  (R = 1, IR end fixed)
    integ = L / zs**2 * zs                        # d(ln z) measure
    area = np.array([np.trapz(integ[i:], lz[i:]) for i in range(len(zs))])
    S_ren = area - I_ct(zs)
    S_local = area - L / zs                       # naive local counterterm (-R*Length of a smooth curve)

    print(f"\n[ren] D_theory={D}, D_fit={Dfit:.3f}, rel. residual={res:.2e}")
    print(f"  smooth pole c_s={c_s:.4f}, leading c_0={c0:.4f}")
    for k, (Ck, sk) in enumerate(zip(C, s), 1):
        print(f"  complex dimension s_{k} = {sk.real:.3f} {sk.imag:+.3f}i :  |C_{k}/c_0| = {abs(Ck) / c0:.3e}")
    win = zs < 0.05
    print(f"  S_ren drift in window (max-min) = {np.ptp(S_ren[win]):.3e}  vs  change of bare area {np.ptp(area[win]):.3e}")
    print(f"  naive local counterterm residual: {S_local[0]:.3e} (z_c={zs[0]:.1e}) -> still divergent, coefficient ~(1/D-1)c_0 z_c^-D")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    u = np.log(zs) / np.log(b)
    Psi = (L - c_s) * zs**(Dfit - 1)
    sc = ax[0].scatter(np.mod(u, 1), Psi / c0, c=u, s=8, cmap='viridis')
    ax[0].set_xlabel("frac(log_b z)"); ax[0].set_ylabel("P(ln z) = (L - c_s) z^{D-1} / c_0")
    ax[0].set_title("Log-periodic function P: periods folded onto each other")
    fig.colorbar(sc, ax=ax[0], label="log_b z")
    ax[1].plot(u, area * zs**Dfit, label="Area·z_c^D (contains P~(ln z_c))")
    ax[1].set_xlabel("log_b z_c"); ax[1].set_title("Log-periodic modulation of the bare area"); ax[1].legend()
    ax[2].plot(u, S_ren, label="Area + I_ct (non-local counterterm)")
    ax[2].plot(u, S_local, '--', label="Area - L(z_c)/z_c (local counterterm)")
    ax[2].set_ylim(np.min(S_ren) - 3 * abs(np.ptp(S_ren)) - 1, np.max(S_ren) + 3 * abs(np.ptp(S_ren)) + 1)
    ax[2].set_xlabel("log_b z_c"); ax[2].set_title("Renormalised entropy (units R^2/4G)"); ax[2].legend()
    savefig(fig, "rt_renormalization")
    meta = [f"linearised Weierstrass model: D={D}, b={b}, d={d}, modes={nmodes}",
            f"fit: D_fit={Dfit:.6f}, rel_residual={res:.3e}, c_s={c_s:.8g}, c_0={c0:.8g}"]
    savedata("renormalization_curves", ["z", "L", "area", "S_ren_nonlocal_ct", "S_local_ct"],
             [zs, L, area, S_ren, S_local], meta)
    savedata("renormalization_complex_dims", ["k", "Re_s", "Im_s", "abs_Ck_over_c0"],
             [np.arange(1, M + 1), [x.real for x in s], [x.imag for x in s], [abs(x) / c0 for x in C]], meta)


# ---------------------------------------------------------------- 4. multifractals / Renyi cosmic branes
def logsumexp(v):
    v = np.asarray(v); mx = np.max(v)
    return mx + np.log(np.sum(np.exp(v - mx)))


def binomial_tau(q, p, r):
    """Binomial two-scale Cantor measure:  Σ_i p_i^q r_i^{-τ} = 1"""
    lp, lr = np.log(p), np.log(r)
    tau = brentq(lambda t: logsumexp(q * lp - t * lr), -500, 500)
    w = np.exp(q * lp - tau * lr)
    alpha = np.sum(w * lp) / np.sum(w * lr)
    return tau, alpha, q * alpha - tau


def cascade_spectrum(p, r, eps):
    """
    Exact cascade 'entanglement spectrum' at resolution eps; stopping rule = first prefix of size <= eps.
    Returns (ln eigenvalue, ln multiplicity), grouped by composition (k1,k2); no 2^n enumeration.
    """
    lp, lr, le = np.log(p), np.log(r), np.log(eps)
    K1, K2 = int(le / lr[0]) + 2, int(le / lr[1]) + 2
    lam, mult = [], []
    for k1 in range(K1 + 1):
        for k2 in range(K2 + 1):
            size = k1 * lr[0] + k2 * lr[1]
            if size > le:
                continue
            terms = []
            if k1 >= 1 and (k1 - 1) * lr[0] + k2 * lr[1] > le:
                terms.append(gammaln(k1 + k2) - gammaln(k1) - gammaln(k2 + 1))
            if k2 >= 1 and k1 * lr[0] + (k2 - 1) * lr[1] > le:
                terms.append(gammaln(k1 + k2) - gammaln(k1 + 1) - gammaln(k2))
            if terms:
                lam.append(k1 * lp[0] + k2 * lp[1]); mult.append(logsumexp(terms))
    return np.array(lam), np.array(mult)


def renyi_from_spectrum(q, lam, mult):
    if abs(q - 1) < 1e-12:
        return -np.sum(np.exp(mult + lam) * lam)
    return logsumexp(mult + q * lam) / (1 - q)


def hyperbolic_brane_x(q, d):
    """
    Cosmic-brane backreaction = hyperbolic black hole f(r) = r^2 - 1 - m/r^{d-2} (R=1).
    Regularity (smooth replica cover <=> deficit 2pi(1-1/q) in the quotient):  T(x_q) = T_0 / q
    T(x)/T_0 = (d x - (d-2)/x) / 2
    """
    return brentq(lambda x: (d * x - (d - 2) / x) / 2 - 1 / q, 1e-8, 1e3)


def renyi_multifractal_demo(p=(0.7, 0.3), r=(0.25, 0.4), eps=1e-60):
    p, r = np.array(p), np.array(r)
    Lg = np.log(1 / eps)
    qs = np.linspace(-6, 8, 141)
    TAF = np.array([binomial_tau(q, p, r) for q in qs])

    # --- check of the Dong identity on the exact spectrum:  S̃_q = q^2 ∂_q((q-1)/q S_q) = S_vN(ρ^q/Trρ^q) = f(α_q)·ln(1/ε)
    lam, mult = cascade_spectrum(p, r, eps)
    print(f"\n[renyi] binomial two-scale Cantor: p={tuple(p)}, r={tuple(r)}, ε={eps:g}")
    print(f"  spectral groups={len(lam)}, sum of eigenvalues = {np.exp(logsumexp(mult + lam)):.12f}")
    print("    q     τ(q)     α_q     f(α_q)   S~_q/L (num. deriv.)  S_vN(escort)/L")
    h = 1e-4
    G = lambda q: (q - 1) / q * renyi_from_spectrum(q, lam, mult) if abs(q - 1) > 1e-9 else 0.0
    for q in [0.5, 1.5, 2, 3, 5]:
        tau, al, f = binomial_tau(q, p, r)
        St = q**2 * (G(q + h) - G(q - h)) / (2 * h)
        lw = mult + q * lam; lw -= logsumexp(lw)
        Sesc = -np.sum(np.exp(lw) * (lw - mult))
        print(f"  {q:4.1f}  {tau:+.4f}  {al:.4f}  {f:.4f}     {St / Lg:.4f}          {Sesc / Lg:.4f}")

    # --- entanglement-spectrum histogram -> f(alpha)
    al_i = -lam / Lg
    bins = np.linspace(al_i.min(), al_i.max(), 120)
    idx = np.digitize(al_i, bins)
    a_h, f_h = [], []
    for j in np.unique(idx):
        sel = idx == j
        a_h.append(np.mean(al_i[sel])); f_h.append(logsumexp(mult[sel]) / Lg)

    # --- holographic (Einstein) cosmic branes
    qh = np.linspace(0.2, 10, 400)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax[0, 0].plot(TAF[:, 1], TAF[:, 2], 'k', lw=2, label="Legendre: f(α)=qα-τ")
    ax[0, 0].plot(a_h, f_h, '.', color='tab:red', ms=4, label=f"entanglement-spectrum histogram (eps={eps:g})")
    ax[0, 0].set_xlabel("α"); ax[0, 0].set_ylabel("f(α)"); ax[0, 0].legend()
    ax[0, 0].set_title("Singularity spectrum of the binomial two-scale Cantor measure")
    ax[0, 1].plot(qs, TAF[:, 0], label="τ(q)")
    ax[0, 1].plot(qs, TAF[:, 2], label="f(α_q) = Area(brane_q)/(4G·lnε⁻¹)")
    ax[0, 1].axhline(0, color='grey', lw=0.5); ax[0, 1].set_xlabel("q"); ax[0, 1].legend()
    ax[0, 1].set_title("Cascade model: mass exponent and brane area")

    meta = [f"binomial two-scale Cantor measure: p={tuple(p)}, r={tuple(r)}"]
    savedata("binomial_cantor_tau_alpha_f", ["q", "tau", "alpha", "f"],
             [qs, TAF[:, 0], TAF[:, 1], TAF[:, 2]], meta)
    savedata("binomial_cantor_spectrum_histogram", ["alpha", "f_hist"], [a_h, f_h],
             meta + [f"exact cascade spectrum at eps={eps:g}"])

    brane_cols, brane_names = [qh], ["q"]
    print("\n  holographic hyperbolic-black-hole branes: check q²∂_q((q-1)/q S_q)/ℒ  =  x_q^{d-1}")
    for d, col in zip([2, 3, 4], ['tab:blue', 'tab:orange', 'tab:green']):
        x = np.array([hyperbolic_brane_x(q, d) for q in qh])
        Sq = np.where(np.abs(qh - 1) < 1e-9, 1.0,
                      qh / (2 * (qh - 1) + 1e-300) * (2 - x**(d - 2) * (1 + x**2)))
        tau = (qh - 1) * Sq
        alpha = np.gradient(tau, qh)
        f = qh * alpha - tau
        def G_(q):   # (q-1)/q · S_q/ℒ,  HMSY closed form
            xq = hyperbolic_brane_x(q, d)
            return (2 - xq**(d - 2) * (1 + xq**2)) / 2
        hh = 1e-6
        St = np.array([q**2 * (G_(q + hh) - G_(q - hh)) / (2 * hh) for q in qh])
        print(f"    d={d}: max|S̃_q/ℒ - x_q^(d-1)| = {np.max(np.abs(St - x**(d - 1))):.2e}")
        brane_cols += [x, Sq, alpha, f]
        brane_names += [f"x_q_d{d}", f"Sq_over_S1_d{d}", f"alpha_d{d}", f"f_d{d}"]
        ax[1, 0].plot(qh, x, color=col, label=f"d={d}")
        ax[1, 1].plot(alpha[3:-3], f[3:-3], color=col, label=f"CFT_{d} (holographic)")
        if d == 2:   # Calabrese-Lefevre analytic spectrum (normalised c/3 = 1)
            aa = np.linspace(0.5, alpha.max(), 200)
            ax[1, 1].plot(aa, np.sqrt(2 * aa - 1), 'k:', label="Calabrese–Lefevre √(2α-1)")
    ax[1, 0].set_xlabel("q"); ax[1, 0].set_ylabel("x_q = r_h/R"); ax[1, 0].legend()
    ax[1, 0].set_title("Brane backreaction: hyperbolic horizon radius")
    ax[1, 1].set_xlabel("alpha (units of S_1)"); ax[1, 1].set_ylabel("f(α)"); ax[1, 1].legend()
    ax[1, 1].set_xlim(0, 4); ax[1, 1].set_ylim(0, 4)
    ax[1, 1].set_title("'Singularity spectrum' of holographic entanglement")

    # cone angle: total angle 2pi/q at the brane in B_q/Z_q
    a = ax[0, 2]
    for i, q in enumerate([1, 1.5, 2, 4]):
        th = np.linspace(0, 2 * np.pi / q, 100)
        cx = 2.4 * i
        a.fill(np.r_[cx, cx + np.cos(th)], np.r_[0, np.sin(th)], alpha=0.35)
        a.text(cx, -1.35, f"q={q}\nΔθ=2π(1-1/q)\n={2 * np.pi * (1 - 1 / q):.2f}", ha='center', fontsize=8)
    a.set_aspect('equal'); a.axis('off'); a.set_title("Conical defect at the brane (tension T_q=(q-1)/4qG)")

    # normalised f(alpha): binomial cascade vs holographic d=2
    x2 = 1 / qh
    tau2 = (qh - 1) * (1 + 1 / qh) / 2
    al2 = (1 + 1 / qh**2) / 2
    ax[1, 2].plot((TAF[:, 1] - TAF[:, 1].min()) / np.ptp(TAF[:, 1]), TAF[:, 2] / TAF[:, 2].max(),
                  'k', label="binomial Cantor (bounded alpha support)")
    sel = al2 < 3
    ax[1, 2].plot((al2[sel] - 0.5) / 2.5, (x2 / 2)[sel] / (x2 / 2)[sel].max(), color='tab:blue',
                  label="holographic CFT2 (unbounded tail)")
    ax[1, 2].set_xlabel("normalised alpha"); ax[1, 2].set_ylabel("normalised f"); ax[1, 2].legend()
    ax[1, 2].set_title("Shape comparison: cascade vs Einstein gravity")
    savefig(fig, "renyi_multifractal")
    savedata("hyperbolic_branes", brane_names, brane_cols,
             ["Einstein gravity, hyperbolic black-hole cosmic branes (R=1); alpha, f in units of S_1"])


# ================================================================ stage 3
# ---------------------------------------------------------------- 5. DSI locking and the Re(s)=0 criterion
def moran_poles(m, k, b, M=30, re_min=-1e-9):
    """Roots of the lattice Moran equation 1 - Σ_j m_j b^{-k_j s} = 0 (polynomial roots in x = b^{-s}, plus 2 pi i m/ln b)"""
    K = max(k)
    c = np.zeros(K + 1)
    for mj, kj in zip(m, k):
        c[K - kj] += mj
    c[K] -= 1.0
    out = []
    for x in np.roots(c):
        s0 = -np.log(complex(x)) / np.log(b)
        for mm in range(-M, M + 1):
            s = s0 + 2j * np.pi * mm / np.log(b)
            if s.real > re_min:
                out.append(s)
    return np.array(out)


def renewal_entropy(m, k, b, eps):
    """
    Exactly solvable toy: L(z) = Σ_words r_w g(z/r_w),  g(u) = u e^{-u}  (generator has only real poles s=0,-1,...)
    S(ε) = ∫_ε^1 z^{-2} L dz = Σ_K c_K [E1(ε b^K) - E1(b^K)],  c_K = number of words with total exponent K
    """
    from scipy.special import exp1
    Kmax = int(np.log(60 / eps.min()) / np.log(b)) + 2
    cK = np.zeros(Kmax + 1); cK[0] = 1.0
    for K in range(1, Kmax + 1):
        cK[K] = sum(mj * cK[K - kj] for mj, kj in zip(m, k) if K - kj >= 0)
    Ks = np.arange(Kmax + 1)
    return np.array([np.sum(cK * (exp1(e * b**Ks) - exp1(b**Ks))) for e in eps])


def renewal_prediction(m, k, b, eps, M=30):
    """Mellin prediction: ζ_L = Γ(s)/M(s).  returns (divergent part, Re s=0 oscillation, log coefficient)"""
    from scipy.special import gamma as cgamma
    Mfun = lambda s: 1 - sum(mj * b**(-kj * s) for mj, kj in zip(m, k))
    dM = lambda s: sum(mj * kj * np.log(b) * b**(-kj * s) for mj, kj in zip(m, k))
    div = np.zeros_like(eps); osc = np.zeros_like(eps)
    for w in moran_poles(m, k, b, M):
        c = cgamma(w) / dM(w)                      # L ∋ c z^{1-w}
        term = np.real(c * (eps**(-w) - 1) / w)    # ∫_ε^1 z^{-1-w} dz
        if w.real > 1e-9:
            div += term
        elif abs(w) > 1e-9:
            osc += term
    lam = 1 / Mfun(0)                              # pole of Gamma at s=0 -> L contains z/M(0) -> logarithm
    return div, osc, lam


def dsi_demo():
    print("\n[stage 3-1] complex dimensions of lattice Moran equations")
    cases = {"Koch type: 4 copies of ratio 1/3": ([4], [1], 3.0),
             "lattice string: ratios 2^-1, 2^-5": ([1, 1], [1, 5], 2.0)}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for i, (name, (m, k, b)) in enumerate(cases.items()):
        P = moran_poles(m, k, b, M=3, re_min=-0.5)
        ax[0].scatter(P.real, P.imag, s=25, label=name, marker='o' if i == 0 else 'x')
        Ds = sorted({round(p.real, 6) for p in P})
        print(f"  {name}: Re(s) ∈ {Ds},  Im spacing 2π/ln b = {2 * np.pi / np.log(b):.4f}")
        eps = np.logspace(-6 if i == 0 else -12, -2, 700)
        S = renewal_entropy(m, k, b, eps)
        div, osc, lam = renewal_prediction(m, k, b, eps)
        resid = S - div - lam * np.log(1 / eps)
        c0 = np.median(resid - osc)
        u = np.log(eps) / np.log(b)
        ax[1 + i].plot(u, resid - c0, lw=2, label="numerics: S - divergent poles - lambda ln(1/eps)")
        ax[1 + i].plot(u, osc, 'k--', lw=1, label="prediction: sum_{Re w=0} c_w eps^{-w}/w")
        ax[1 + i].set_xlabel("log_b ε"); ax[1 + i].set_title(f"{name}\nlog coefficient lambda = 1/M(0) = {lam:.4f}")
        ax[1 + i].legend(fontsize=8)
        print(f"    S_ren residual minus prediction, max deviation = {np.max(np.abs(resid - c0 - osc)):.2e},"
              f"  Re=0 oscillation amplitude = {np.ptp(osc):.3e}")
        tag = "koch" if i == 0 else "lattice_string_1_5"
        meta = [f"{name}: m={m}, k={k}, b={b}", f"log coefficient lambda=1/M(0)={lam:.10g}"]
        savedata(f"moran_poles_{tag}", ["Re_s", "Im_s"], [P.real, P.imag], meta)
        savedata(f"marginal_residual_{tag}", ["log_b_eps", "residual", "prediction_Re0"],
                 [u, resid - c0, osc], meta)
    ax[0].axvline(0, color='grey', lw=0.6)
    ax[0].set_xlabel("Re s"); ax[0].set_ylabel("Im s"); ax[0].legend(fontsize=8)
    ax[0].set_title("Complex dimensions (Re s>0 divergent, Re s=0 marginal)")
    savefig(fig, "dsi_complex_dims")


# ---------------------------------------------------------------- 6. complex mass exponents: period locked in q
def multifractal_lockin_demo(p=(0.7, 0.3)):
    p = np.array(p)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for j, (r, title) in enumerate([((0.5, 0.25), "Lattice r=(1/2,1/4): period ln2 for all q"),
                                    ((0.25, 0.4), "Non-lattice r=(0.25,0.4): no strict period")]):
        r = np.array(r)
        eps = np.logspace(-4, -16, 400)
        cols, names = [np.log2(eps)], ["log2_eps"]
        for q in [-2, 0, 2, 4]:
            tau = binomial_tau(q, p, r)[0]
            y = []
            for e in eps:
                lam, mult = cascade_spectrum(p, r, e)
                y.append(logsumexp(mult + q * lam) - tau * np.log(e))
            y = np.array(y) - np.mean(y)
            ax[j].plot(np.log2(eps), y, label=f"q={q}, τ={tau:+.3f}")
            cols.append(y); names.append(f"q{q}")
        ax[j].set_xlabel("log_2 ε"); ax[j].set_ylabel("ln Z_q(ε) - τ(q) ln ε")
        ax[j].set_title(title); ax[j].legend(fontsize=8)
        savedata(f"multifractal_lockin_{'lattice' if j == 0 else 'nonlattice'}", names, cols,
                 [f"ln Z_q(eps) - tau(q) ln eps (mean removed); p={tuple(p)}, r={tuple(r)}"])
    savefig(fig, "multifractal_lockin")


# ---------------------------------------------------------------- 7. geodesic bit-thread flow
def verify_geodesic_threads():
    x, z, R, G = sp.symbols('x z R G', positive=True)
    d = sp.symbols('d', positive=True, integer=True)
    u = sp.sqrt(x**2 + z**2)
    sing = z / u
    vx = sing**(d - 1) / (4 * G) * (z / R) * (z / u)      # tangent to semicircles x^2+z^2=u^2
    vz = sing**(d - 1) / (4 * G) * (z / R) * (-x / u)
    sqrtg = (R / z)**d                                      # volume element of the H^d slice
    div = sp.simplify(sp.diff(sqrtg * vx, x) + sp.diff(sqrtg * vz, z))
    norm = sp.simplify(sp.sqrt((R / z)**2 * (vx**2 + vz**2)))
    print("\n[stage 3-3] geodesic bit-thread flow (planar entangling surface)")
    print("  ∇·v =", div)
    print("  |v|  =", norm, "  (<= 1/4G, saturated only on the RT surface x=0)")


# ================================================================ stage 4
# ---------------------------------------------------------------- 8. BCFT Schmidt weights -> p_j, and the d=2 consistency check
def bcft_closure_demo(c=12.0, W=400.0):
    """
    Cardy-Tonni: open channel of the annulus of width W=2ln(l/eps):  λ_h = exp[-(2π²/W)(h - c/24)] / Z
    (a) dense Cardy spectrum ρ(E) ~ exp(2π√(cE/6)) => f(α) = √((c/3)(2α - c/3)) = holographic cosmic-brane result
    (b) sparse spectrum (Ising channels Delta in {0, 1/16, 1/2}) => non-trivial p_j ∝ exp(-(2π²/λ)Δ_j)
    """
    L = W / 2
    E = np.linspace(1e-6, 40 * W**2 / (4 * np.pi**2), 400000)
    lnrho = 2 * np.pi * np.sqrt(c * E / 6) + np.log(E[1] - E[0])
    lnlam_u = -(2 * np.pi**2 / W) * E                      # unnormalised (the c/24 shift is absorbed by Z)
    lnZ = logsumexp(lnrho + lnlam_u)
    qs = np.linspace(0.3, 6, 300)
    Sq = np.array([(logsumexp(lnrho + q * lnlam_u) - q * lnZ) / (1 - q) if abs(q - 1) > 1e-9 else np.nan
                   for q in qs])
    ok = ~np.isnan(Sq)
    qs, Sq = qs[ok], Sq[ok]
    tau = (qs - 1) * Sq / L
    al = np.gradient(tau, qs); f = qs * al - tau
    f_cl = np.sqrt(np.clip((c / 3) * (2 * al - c / 3), 0, None))
    print(f"\n[stage 4-1] dense Cardy spectrum -> f(alpha): max deviation from √((c/3)(2α-c/3)) / (c/6) = "
          f"{np.max(np.abs(f - f_cl)[5:-5]) / (c / 6):.2e}  (c={c}, W={W}; finite-W corrections ~ ln W / W)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax[0].plot(al / (c / 6), f / (c / 6), lw=3, alpha=0.5, label="BCFT open channel + Cardy density (numerical)")
    ax[0].plot(al / (c / 6), f_cl / (c / 6), 'k--', label="cosmic brane x_q=1/q (analytic)")
    ax[0].set_xlabel("α / (c/6)"); ax[0].set_ylabel("f / (c/6)"); ax[0].legend()
    ax[0].set_title("d=2 check: Schmidt spectrum = holographic singularity spectrum")

    savedata("bcft_cardy_spectrum", ["q", "alpha_over_c6", "f_over_c6", "f_CL_over_c6"],
             [qs, al / (c / 6), f / (c / 6), f_cl / (c / 6)], [f"c={c}, W={W}"])
    b = 3.0
    Delta = np.array([0.0, 1 / 16, 1 / 2])
    print("  Ising-channel cascade (b=3, annulus width per generation = ln b):")
    for lam_w, ls in [(np.log(b), '-'), (4 * np.log(b), '--')]:
        pj = np.exp(-(2 * np.pi**2 / lam_w) * Delta); pj /= pj.sum()
        r = np.full(3, 1 / b)
        T = np.array([binomial_tau(q, pj, r) for q in np.linspace(-8, 8, 161)])
        print(f"    W_gen={lam_w:.3f}: p_j = {np.round(pj, 5)}, D_0={-binomial_tau(0, pj, r)[0]:.3f}, D_1={T[np.argmin(np.abs(np.linspace(-8, 8, 161) - 1)), 1]:.3f}")
        ax[1].plot(T[:, 1], T[:, 2], ls, label=f"W_gen = {lam_w:.2f}: p={np.round(pj, 3)}")
        savedata(f"ising_cascade_Wgen_{lam_w:.3f}", ["q", "tau", "alpha", "f"],
                 [np.linspace(-8, 8, 161), T[:, 0], T[:, 1], T[:, 2]],
                 [f"Ising channels Delta={tuple(Delta)}, b={b}, W_gen={lam_w:.6f}, p_j={tuple(np.round(pj, 8))}"])
    ax[1].set_xlabel("α"); ax[1].set_ylabel("f(α)"); ax[1].legend(fontsize=8)
    ax[1].set_title("Sparse spectrum (Ising boundary operators) -> non-trivial multifractality")
    savefig(fig, "bcft_closure")


# ---------------------------------------------------------------- 9. fractal-string resonator: Re s=0 fingerprint
def resonator_prediction_demo(v_ph=1.2e8, ell0=0.10):
    """
    Lattice fractal string: generator = 1 copy of ratio 1/3 + 2 copies of ratio 1/9.  Moran: 2x²+x-1=(2x-1)(x+1), x=3^{-s}
      x=1/2 -> D = ln2/ln3;   x=-1 -> Re s = 0, Im s = (2m+1)π/ln3   (half-integer log-harmonics)
    Heat-kernel-smoothed mode count Θ(u) = Σ_w Σ_k exp(-k²/(u ℓ_w)²),  u = 2 ℓ_0 f / v_ph
    Mellin:  Θ̃(s) = ½ Γ(s/2) ζ(s) ζ_L(s),  ζ_L = 1/(1 - 3^{-s} - 2·9^{-s})
    """
    import mpmath as mp
    b = 3.0
    Kmax = 200
    cK = np.zeros(Kmax + 1); cK[0] = 1; cK[1] = 1
    for K in range(2, Kmax + 1):
        cK[K] = cK[K - 1] + 2 * cK[K - 2]
    lnc = np.log(cK)

    def Rv(v):                                  # H(v) - sqrt(pi) v/2, computed without cancellation
        if v < 1:
            k = np.arange(1, 12)
            return np.sum(np.exp(-k**2 / v**2)) - np.sqrt(np.pi) * v / 2
        n = np.arange(1, 12)
        return -0.5 + np.sqrt(np.pi) * v * np.sum(np.exp(-np.pi**2 * n**2 * v**2))

    us = np.logspace(1, 14, 1500)
    Th = np.array([sum(np.exp(lnc[K]) * Rv(u * b**(-K)) for K in range(Kmax + 1)) for u in us])

    M1 = lambda s: mp.log(b) * (b**(-s) + 4 * b**(-2 * s))
    D = np.log(2) / np.log(b)
    def fam(s0, mrange):
        out = np.zeros_like(us)
        for m in mrange:
            s = mp.mpc(s0, 2 * mp.pi * m / mp.log(b)) if s0 > 0 else mp.mpc(0, (2 * m + 1) * mp.pi / mp.log(b))
            A = complex(0.5 * mp.gamma(s / 2) * mp.zeta(s) / M1(s))
            out += np.real(A * us**complex(s))
        return out, A
    Dfam, _ = fam(D, range(-25, 26))
    Re0, _ = fam(0, range(-25, 25))
    const = 0.25                                 # ½·Res Γ(s/2)|_0 · ζ(0)/M(0) = ½·2·(-½)/(-2)
    resid = Th - Dfam - const
    print(f"\n[stage 4-2] fractal-string resonator: D = {D:.4f}")
    print(f"  Theta minus Weyl, D family and constant vs Re=0 prediction, max deviation = {np.max(np.abs(resid - Re0)):.2e}")
    print(f"  Re=0 peak-to-peak = {np.ptp(Re0):.4f} modes;  D-family oscillation/power ~ {np.ptp(Dfam / us**D):.3f}")
    A1 = complex(0.5 * mp.gamma(1j * mp.pi / mp.log(b) / 2) * mp.zeta(1j * mp.pi / mp.log(b)) / M1(1j * mp.pi / mp.log(b)))
    print(f"  fundamental amplitude |A_0| = {abs(A1):.4f}, log-period = 2 ln3 (repeats every factor 9 in frequency)")
    f1 = v_ph / (2 * ell0)
    print(f"  example device (ℓ_0={ell0} m, v={v_ph:.1e} m/s): fundamental {f1 / 1e9:.2f} GHz; two Re=0 periods need f in "
          f"[{f1 / 1e9:.2f}, {81 * f1 / 1e9:.1f}] GHz;  shortest branch ℓ_0·3^-4 = {ell0 / 81 * 1e3:.2f} mm")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    u3 = np.log(us) / np.log(b)
    ax[0].plot(u3, (Th - const) / us**D, label="Θ_res / u^D (contains D family)")
    ax[0].set_xlabel("log_3 u"); ax[0].set_title("D family: integer log-harmonics (period ln3)"); ax[0].legend()
    ax[1].plot(u3, resid, lw=2.5, alpha=0.6, label="numerics: Theta - Weyl - D family - 1/4")
    ax[1].plot(u3, Re0, 'k--', label="Mellin prediction (Re s=0)")
    ax[1].set_xlabel("log_3 u"); ax[1].set_ylabel("number of modes"); ax[1].legend()
    ax[1].set_title("Re s=0 family: bounded oscillation, period 2ln3")
    # log-frequency spectrum
    grid = np.linspace(u3[0], u3[-1], 4096)
    for y, lab in [((Th - const) / us**D, "D family (divided by u^D)"), (resid, "Re=0 family")]:
        yy = np.interp(grid, u3, y); yy = (yy - yy.mean()) * np.hanning(len(yy))
        F = np.abs(np.fft.rfft(yy)); fr = np.fft.rfftfreq(len(yy), grid[1] - grid[0])
        ax[2].plot(fr, F / F.max(), label=lab)
    ax[2].set_xlim(0, 3.2); ax[2].set_xlabel("cycles per log_3 unit")
    for x in np.arange(0, 3.5, 0.5):
        ax[2].axvline(x, color='grey', lw=0.4, ls=':')
    ax[2].set_title("Fingerprint: integer (D) vs half-integer (Re s=0)"); ax[2].legend()
    savefig(fig, "resonator_fingerprint")
    savedata("resonator_heat_trace", ["log3_u", "Theta_minus_Weyl", "D_family", "residual", "prediction_Re0"],
             [u3, Th, Dfam, resid, Re0],
             ["lattice fractal string: 1 copy ratio 1/3 + 2 copies ratio 1/9; u = 2 l_0 f / v",
              f"D={D:.10g}; constant term 1/4; fundamental |A_0|={abs(A1):.6g}"])


def main_stage1():
    verify_curvature()

    d = 3
    Ds = [1.1, 1.3, 1.5, 1.7, 1.9]
    zs = np.logspace(-3.2, 0.3, 60)
    zmax = zs[-1]
    fit = (zs > 3 / 3**5) & (zs < 0.3)   # scaling window: 1/k_max << z << 1/k0

    # figure 1: profile "streamlines"
    fig, axes = plt.subplots(2, len(Ds), figsize=(4 * len(Ds), 7.5), constrained_layout=True)
    zshow = [0.002, 0.01, 0.04, 0.15, 0.6]
    cmap = plt.get_cmap('viridis')
    for j, D in enumerate(Ds):
        m = FractalRT(D, d=d)
        sl = m.y < 2 * np.pi / 3
        for i, z in enumerate(zshow):
            f, _, _ = m.fields(z)
            axes[0, j].plot(m.y[sl], f[sl], color=cmap(i / (len(zshow) - 1)), lw=0.8, label=f"z={z}")
        axes[0, j].set_title(f"D = {D}: cross-sections x=f(y,z)")
        axes[0, j].set_xlabel("y")
        if j == 0:
            axes[0, j].set_ylabel("x"); axes[0, j].legend(fontsize=7)
        # (y, z) plane: level sets of f = profile streamlines of the embedding
        yy = m.y[sl][::8]
        zz = np.logspace(-3, 0, 120)
        F = np.array([m.fields(z)[0][sl][::8] for z in zz])
        axes[1, j].contour(yy, zz, F, levels=25, linewidths=0.6, cmap='coolwarm')
        axes[1, j].set_yscale('log'); axes[1, j].invert_yaxis()
        axes[1, j].set_xlabel("y")
        if j == 0:
            axes[1, j].set_ylabel("z (resolution; down = towards boundary)")
    savefig(fig, "rt_profiles")

    # figure 2: divergence exponents
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    print(f"\n d={d}  theory: L(ε)~ε^(1-D),  Area(ε)~ε^(-(D+d-3))")
    print("   D    L exp(num)   L exp(th)     Area exp(num)   Area exp(th)")
    table, Lcols, Acols = [], [zs], [zs]
    for D in Ds:
        m = FractalRT(D, d=d)
        L = np.array([m.area_density(z) for z in zs])
        # Area(ε) = R^{d-1} ∫_ε^{zmax} z^{1-d} L(z) dz   (R=1, flat-direction length = 1)
        integrand = zs**(1 - d) * L
        lz = np.log(zs)
        cum = np.array([np.trapz((integrand * zs)[i:], lz[i:]) for i in range(len(zs))])
        sL = np.polyfit(np.log(zs[fit]), np.log(L[fit]), 1)[0]
        sA = np.polyfit(np.log(zs[fit]), np.log(cum[fit]), 1)[0]
        print(f"  {D:.1f}     {sL:+.3f}       {1 - D:+.3f}        {sA:+.3f}         {-(D + d - 3):+.3f}")
        table.append([D, sL, 1 - D, sA, -(D + d - 3)])
        Lcols.append(L); Acols.append(cum)
        ax1.loglog(zs, L, label=f"D={D}")
        ax2.loglog(zs, cum, label=f"D={D}")
    for ax in (ax1, ax2):
        ax.axvspan(zs[fit][0], zs[fit][-1], color='grey', alpha=0.12)
        ax.set_xlabel("ε = z"); ax.legend(fontsize=8)
    ax1.set_title("Cross-section length L(z)  (~ z^{1-D})")
    ax2.set_title("RT area Area(eps) ~ 4G_N S_A  (~ eps^{-D})")
    savefig(fig2, "rt_scaling")
    meta = [f"linearised Weierstrass model, d={d}, b=3, six modes; fit window {zs[fit][0]:.4g} <= z <= {zs[fit][-1]:.4g}"]
    T = np.array(table)
    savedata("table1_exponents", ["D", "content_exp_num", "content_exp_theory", "entropy_exp_num", "entropy_exp_theory"],
             [T[:, i] for i in range(5)], meta)
    savedata("content_vs_z", ["z"] + [f"L_D{D}" for D in Ds], Lcols, meta)
    savedata("regulated_area_vs_z", ["z"] + [f"Area_D{D}" for D in Ds], Acols, meta)


if __name__ == "__main__":
    import sys
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    what = args[0] if args else "all"
    if what not in MODES:
        sys.exit(f"unknown mode '{what}'; choose one of: {', '.join(MODES)}")
    if what in ("rt", "all"):
        main_stage1()
    if what in ("ren", "all"):
        renormalization_demo()
    if what in ("renyi", "all"):
        renyi_multifractal_demo()
    if what in ("stage3", "all"):
        dsi_demo()
        multifractal_lockin_demo()
        verify_geodesic_threads()
    if what in ("stage4", "all"):
        bcft_closure_demo()
        resonator_prediction_demo()
    if "--show" in sys.argv:
        plt.show()
