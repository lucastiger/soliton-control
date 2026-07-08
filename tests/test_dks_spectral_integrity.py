"""Spectral-integrity checks V1-V6 for the single-DKS pipeline spectrum.

These pin the *shape* of the validated single-soliton comb against the physics
it must contain, and in particular catch numerics that amputate real spectrum
(the failure mode of the old |D_int*t_r|-keyed dispersion-validity mask, which
carved holes into the soliton tail at |mu| ~ 1000-3000):

  V1  the sech tail is linear in dB over 400 <= |mu| <= 1500 on each side
      (RMS < 3 dB) with red/blue slope asymmetry < 15%;
  V2  NO interior hole: everywhere in 300 < |mu| < 2900 (excluding +/-80 modes
      around each far-detuned D_int = delta_omega crossing) the spectrum stays
      within 25 dB of that side's fitted tail line — this is the check that
      catches validity-mask-type amputation;
  V3  sech^2 core: a dB-space sech^2 fit over 3 <= |mu| <= 200 has RMS
      residual < 2 dB;
  V4  dispersive-wave peaks within +/-60 modes of both phase-matched crossings,
      prominence > 50 dB above the sech-tail baseline;
  V5  median numerical floor beyond |mu| = n_tau/3 + 50 (past the 2/3 dealias
      cutoff) below -300 dB;
  V6  stationarity: the 8-kappa attractor is a deterministic BREATHER
      (established by controlled numpy-vs-jax A/B, thermal/noise off, parabola
      control): U_int oscillates with period T_b ~ 152-153 RT and rel-std
      ~4.1% > 0.5%, so the state must be classified "breather", never a
      stationary single soliton.

Because the state breathes, every SNAPSHOT spectrum is breathing-phase
dependent — V1-V5 previously passed only on lucky snapshot phases. They
therefore run on the CYCLE-AVERAGED spectrum (mean of |fftshift(fft(E))|^2
accumulated every round trip over CYCLE_AVG_RT = 304 RT >= 2*T_b), which is
deterministic and phase-independent. V6 also pins the cycle-averaged
reference levels at this operating point (red tail slope -0.042+/-0.003
dB/mode; -78+/-3 dB @ 1800 nm; -103+/-3 dB @ 2000 nm; DWs ~-88.5 dB @ 2529 nm
and ~-91 dB @ 1096 nm).

Operating point: the Step-3 validated soliton — delta_omega = 8 kappa,
pin = 0.214 W, seeded single soliton, n_tau = 16384, ~12000 round trips + 304
averaging round trips, with the production numerics stack (float64,
n_substeps = 4, 2/3 dealias ON, edge absorber ON, dispersion-validity mask
OFF).

This is a slow test (one long solver run, shared by all checks through a
module-scoped fixture): a few minutes on CPU. It is the regression harness for
the pipeline spectrum, not a CI-cheap unit test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import curve_fit

from analysis.dks_access import (
    CYCLE_AVG_RT_8KAPPA,
    PIN_W,
    PRODUCTION_NUMERICS,
    access_by_seeding,
    attach_dispersion,
    cycle_averaged_spectrum,
    dispersive_wave_crossings,
    dispersive_wave_peaks,
    load_cavity_params,
)

N_TAU = 16384
N_ROUND_TRIPS = 12_000     # ~0.1 tau_th: seed fully relaxed onto the attractor
DW_KAPPA = 8.0
CYCLE_AVG_RT = CYCLE_AVG_RT_8KAPPA   # 304 RT >= 2*T_b (T_b ~ 152-153 RT)

TAIL_MU_LO, TAIL_MU_HI = 400, 1500
HOLE_MU_LO, HOLE_MU_HI = 300, 2900
HOLE_EXCLUDE_HALF_WIDTH = 80   # modes excluded around each far crossing
CORE_MU_LO, CORE_MU_HI = 3, 200
DW_SCAN_MODES = 60
FLOOR_MARGIN_MODES = 50

# V6 breather references (controlled A/B at 8 kappa, measured D_int grid).
V6_PERIOD_RT = (140.0, 165.0)     # established T_b ~ 152-153 RT
V6_RELSTD = (0.03, 0.05)          # established dU/U ~ 4.08%
REF_RED_TAIL_SLOPE = (-0.042, 0.003)          # dB/mode +/- tol
REF_DB_AT_NM = {1800.0: (-78.0, 3.0), 2000.0: (-103.0, 3.0)}
REF_DW_RED = (2529.0, -88.5, 3.0)             # (nm, dB, tol)
REF_DW_BLUE = (1096.0, -91.0, 3.0)


@pytest.fixture(scope="module")
def soliton_run():
    """One long production-numerics run shared by all checks.

    The settling run is followed by a CYCLE_AVG_RT-round-trip continuation
    whose per-round-trip spectra are averaged; V1-V5 run on that
    cycle-averaged spectrum (snapshot spectra of the breather are
    phase-dependent).
    """
    cav = load_cavity_params()
    cav = attach_dispersion(cav, N_TAU)
    dw = DW_KAPPA * cav.kappa
    res = access_by_seeding(dw, cav, t_slow=N_ROUND_TRIPS, seed=0, n_tau=N_TAU,
                            pin=PIN_W, **PRODUCTION_NUMERICS)
    # Precondition: every check below is about THIS state's spectrum, so the
    # run must actually be a validated single soliton (the existence label
    # stands even though the state breathes — see V6).
    assert res["is_single"], (
        f"pipeline run did not converge to a single soliton: {res['metrics']}"
    )
    sp = cycle_averaged_spectrum(res, cav, n_rt=CYCLE_AVG_RT, pin=PIN_W,
                                 **PRODUCTION_NUMERICS)
    return {"cav": cav, "dw": dw, "res": res, "sp": sp}


def _tail_fit(sp: dict, side: int) -> dict:
    """Line fit (dB vs |mu|) to one side's sech tail over [TAIL_MU_LO, TAIL_MU_HI].

    ``side`` = +1 for the blue side (mu > 0), -1 for the red side (mu < 0).
    Returns slope [dB/mode], intercept, and RMS residual [dB].
    """
    mu = np.asarray(sp["mu"])
    db = np.asarray(sp["power_db"])
    sel = (mu * side >= TAIL_MU_LO) & (mu * side <= TAIL_MU_HI)
    x = np.abs(mu[sel]).astype(np.float64)
    y = db[sel]
    slope, intercept = np.polyfit(x, y, 1)
    rms = float(np.sqrt(np.mean((np.polyval([slope, intercept], x) - y) ** 2)))
    return {"slope": float(slope), "intercept": float(intercept), "rms": rms}


def test_v1_tail_linear_in_db_and_side_symmetric(soliton_run):
    """V1: dB-linear sech tail, RMS < 3 dB per side, slope asymmetry < 15%."""
    sp = soliton_run["sp"]
    blue = _tail_fit(sp, +1)
    red = _tail_fit(sp, -1)
    assert blue["rms"] < 3.0, f"blue tail RMS {blue['rms']:.2f} dB >= 3 dB"
    assert red["rms"] < 3.0, f"red tail RMS {red['rms']:.2f} dB >= 3 dB"
    assert blue["slope"] < 0 and red["slope"] < 0, "tail must decay with |mu|"
    s_b, s_r = abs(blue["slope"]), abs(red["slope"])
    asym = abs(s_b - s_r) / (0.5 * (s_b + s_r))
    assert asym < 0.15, (
        f"red/blue tail-slope asymmetry {asym:.1%} >= 15% "
        f"(blue {blue['slope']:.4f}, red {red['slope']:.4f} dB/mode)"
    )


def test_v2_no_interior_hole(soliton_run):
    """V2: no mode in 300 < |mu| < 2900 falls > 25 dB below the tail line.

    This is the amputation detector: the old |D_int*t_r|-keyed validity mask
    carved the spectrum down to the numerical floor over mu ~ [-2950, -1000]
    and [+1150, +3050], hundreds of dB below the sech-tail extrapolation.
    Upward excursions (dispersive-wave shoulders) are physical and allowed;
    +/-80 modes around each far-detuned D_int = delta_omega crossing are
    excluded.
    """
    sp = soliton_run["sp"]
    mu = np.asarray(sp["mu"])
    db = np.asarray(sp["power_db"])
    crossings = dispersive_wave_crossings(soliton_run["dw"])

    worst = []
    for side in (+1, -1):
        fit = _tail_fit(sp, side)
        sel = (mu * side > HOLE_MU_LO) & (mu * side < HOLE_MU_HI)
        for c in crossings:
            sel &= np.abs(mu - c["crossing_mu"]) > HOLE_EXCLUDE_HALF_WIDTH
        line = fit["slope"] * np.abs(mu[sel]) + fit["intercept"]
        deficit = line - db[sel]          # >0 means below the tail line
        i = int(np.argmax(deficit))
        worst.append((float(deficit[i]), int(mu[sel][i])))
        bad = mu[sel][deficit > 25.0]
        assert bad.size == 0, (
            f"interior hole on the {'blue' if side > 0 else 'red'} side: "
            f"{bad.size} modes fall > 25 dB below the tail line, "
            f"mu in [{bad.min()}, {bad.max()}], worst deficit "
            f"{deficit.max():.1f} dB at mu = {mu[sel][i]}"
        )
    print(f"[V2] worst (deficit_dB, mu) per side: blue {worst[0]}, red {worst[1]}")


def test_v3_sech2_core(soliton_run):
    """V3: dB-space sech^2 fit over 3 <= |mu| <= 200, RMS residual < 2 dB."""
    sp = soliton_run["sp"]
    mu = np.asarray(sp["mu"]).astype(np.float64)
    db = np.asarray(sp["power_db"])
    sel = (np.abs(mu) >= CORE_MU_LO) & (np.abs(mu) <= CORE_MU_HI)
    x, y = mu[sel], db[sel]

    def sech2_db(m, w, c):
        # 10*log10(sech^2(m/w)) + c, via a numerically safe log-cosh
        a = np.abs(m) / w
        log_cosh = a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)
        return c - 20.0 / np.log(10.0) * log_cosh

    # init width from the core's own dB slope at the window edge (~ -8.686/w)
    w0 = 8.686 / max(abs((y[x == x.max()][0] - y.max()) / x.max()), 1e-6)
    popt, _ = curve_fit(sech2_db, x, y, p0=[w0, float(y.max())], maxfev=20000)
    rms = float(np.sqrt(np.mean((sech2_db(x, *popt) - y) ** 2)))
    print(f"[V3] sech^2 core fit: width {popt[0]:.1f} modes, RMS {rms:.2f} dB")
    assert rms < 2.0, f"sech^2 core fit RMS {rms:.2f} dB >= 2 dB"


def test_v4_dispersive_wave_peaks(soliton_run):
    """V4: DW peaks within +/-60 modes of both crossings, prominence > 50 dB."""
    sp, dw = soliton_run["sp"], soliton_run["dw"]
    crossings = dispersive_wave_crossings(dw)
    assert len(crossings) == 2, f"expected 2 far crossings, got {crossings}"
    peaks = dispersive_wave_peaks(sp, dw, scan_modes=DW_SCAN_MODES)
    assert len(peaks) == 2, f"expected 2 DW peaks, got {peaks}"
    sides = sorted(np.sign(p["mu"]) for p in peaks)
    assert sides == [-1.0, 1.0], f"need one DW per side, got {peaks}"
    for p in peaks:
        print(f"[V4] DW at mu = {p['mu']:+d} ({p['wavelength_nm']:.0f} nm): "
              f"{p['power_db']:.1f} dB, prominence {p['prominence_db']:.1f} dB")
        assert abs(p["mu"] - p["crossing_mu"]) <= DW_SCAN_MODES
        assert p["prominence_db"] > 50.0, (
            f"DW at mu = {p['mu']:+d} prominence {p['prominence_db']:.1f} dB "
            f"<= 50 dB"
        )


def test_v5_numerical_floor(soliton_run):
    """V5: median floor beyond |mu| = n_tau/3 + 50 below -300 dB."""
    sp = soliton_run["sp"]
    mu = np.asarray(sp["mu"])
    db = np.asarray(sp["power_db"])
    sel = np.abs(mu) > (N_TAU / 3.0 + FLOOR_MARGIN_MODES)
    floor = float(np.median(db[sel]))
    print(f"[V5] median floor beyond |mu| = {N_TAU / 3 + FLOOR_MARGIN_MODES:.0f}: "
          f"{floor:.1f} dB")
    assert floor < -300.0, f"numerical floor {floor:.1f} dB >= -300 dB"


def test_v6_breather_classification(soliton_run):
    """V6: the 8-kappa state is a breather, NOT a stationary single soliton."""
    m = soliton_run["res"]["metrics"]
    print(f"[V6] stationarity={m['stationarity']}, "
          f"T_b={m['breathing_period_rt']:.1f} RT, "
          f"dU/U={m['breathing_relstd']:.2%}")
    assert m["is_breather"], (
        f"V6 must classify the 8-kappa state as a breather "
        f"(rel-std {m['breathing_relstd']:.2%} expected ~4.1% > 0.5%)"
    )
    assert m["stationarity"] == "breather"
    lo, hi = V6_PERIOD_RT
    assert lo <= m["breathing_period_rt"] <= hi, (
        f"breathing period {m['breathing_period_rt']:.1f} RT outside "
        f"[{lo}, {hi}] (established T_b ~ 152-153 RT)"
    )
    lo, hi = V6_RELSTD
    assert lo <= m["breathing_relstd"] <= hi, (
        f"breathing rel-std {m['breathing_relstd']:.2%} outside "
        f"[{lo:.0%}, {hi:.0%}] (established ~4.08%)"
    )


def _db_at_wavelength(sp: dict, lam_nm: float) -> float:
    i = int(np.argmin(np.abs(np.asarray(sp["wavelength_nm"]) - lam_nm)))
    return float(np.asarray(sp["power_db"])[i])


def test_v6_cycle_averaged_reference_levels(soliton_run):
    """Cycle-averaged spectrum reference levels at 8 kappa (phase-independent).

    These are the deterministic values the committed artifacts must show; a
    snapshot spectrum only matches them at lucky breathing phases.
    """
    sp = soliton_run["sp"]
    assert sp.get("n_rt_averaged") == CYCLE_AVG_RT

    slope = _tail_fit(sp, -1)["slope"]
    ref, tol = REF_RED_TAIL_SLOPE
    print(f"[V6-ref] red tail slope {slope:.4f} dB/mode (ref {ref}±{tol})")
    assert abs(slope - ref) <= tol, (
        f"red tail slope {slope:.4f} dB/mode outside {ref}±{tol}"
    )

    for lam, (ref_db, tol_db) in REF_DB_AT_NM.items():
        db = _db_at_wavelength(sp, lam)
        print(f"[V6-ref] {db:.1f} dB @ {lam:.0f} nm (ref {ref_db}±{tol_db})")
        assert abs(db - ref_db) <= tol_db, (
            f"{db:.1f} dB @ {lam:.0f} nm outside {ref_db}±{tol_db} dB"
        )

    peaks = dispersive_wave_peaks(sp, soliton_run["dw"], scan_modes=DW_SCAN_MODES)
    assert len(peaks) == 2, f"expected 2 DW peaks, got {peaks}"
    for lam_ref, db_ref, tol_db in (REF_DW_RED, REF_DW_BLUE):
        p = min(peaks, key=lambda q: abs(q["wavelength_nm"] - lam_ref))
        print(f"[V6-ref] DW {p['power_db']:.1f} dB @ {p['wavelength_nm']:.0f} nm "
              f"(ref {db_ref}±{tol_db} @ ~{lam_ref:.0f} nm)")
        assert abs(p["wavelength_nm"] - lam_ref) < 30.0, (
            f"DW at {p['wavelength_nm']:.0f} nm too far from ref {lam_ref:.0f} nm"
        )
        assert abs(p["power_db"] - db_ref) <= tol_db, (
            f"DW level {p['power_db']:.1f} dB @ {p['wavelength_nm']:.0f} nm "
            f"outside {db_ref}±{tol_db} dB"
        )
