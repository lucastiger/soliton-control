#!/usr/bin/env python
"""Regenerate the committed DKS artifacts in analysis/results/ (auditable).

The previous two artifact regenerations were ad-hoc, so their parameters could
not be verified from the repo. This script IS the regeneration invocation: the
committed PNG/CSV artifacts must be reproducible by running

    python scripts/regenerate_dks_artifacts.py

from the repo root. Every solver parameter is pinned below (CLI flags exist
only to run cheaper debug variants; the committed artifacts use the defaults).

Stages
------
1. SETTLE   Seeded single-DKS run at DW_KAPPA = 8 kappa, pin = 0.214 W,
            n_tau = 16384, 12000 round trips, seed 0, production numerics
            (float64, n_substeps = 4, 2/3 dealias ON, edge absorber ON,
            dispersion-validity mask OFF).
2. V6       Stationarity check on the U_int history. At 8 kappa the attractor
            is a deterministic BREATHER (controlled A/B: period ~152-153 RT,
            dU/U ~4.1%), so the script aborts if V6 does NOT report a breather
            (that would mean the committed annotation is stale).
3. AVERAGE  Cycle-averaged spectrum: continue the settled trajectory for
            CYCLE_AVG_RT = 304 RT (>= 2 breathing periods), accumulating
            |fftshift(fft(E))|^2 every round trip. Regenerates
            dks_single_soliton_spectrum.png and dks_single_soliton_summary.png
            from the MEAN spectrum, title-annotated with the measured breather
            period / rel-std. Snapshot spectra of a breather are
            breathing-phase-dependent and must not be committed.
4. SCAN     Breathing scan delta_omega = 7..16 kappa (0.5 steps, 4000 RT,
            n_tau = 8192, production numerics): writes
            dks_breathing_scan.csv / .png and merges the V6 breathing columns
            into dks_existence_map.csv (the is_single labels stand; only the
            breathing metadata columns are added/updated).
5. PROVENANCE  Writes dks_artifact_provenance.json recording the exact
            parameters and the key measured values of this regeneration.

Expected reference values at the 8-kappa operating point (cycle average,
n_tau = 16384): red tail slope -0.042+/-0.003 dB/mode, -78+/-3 dB @ 1800 nm,
-103+/-3 dB @ 2000 nm, DWs ~-88.5 dB @ 2529 nm and ~-91 dB @ 1096 nm. These
are pinned by tests/test_dks_spectral_integrity.py (V1-V6).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from analysis.dks_access import (  # noqa: E402  (needs the sys.path insert)
    CYCLE_AVG_RT_8KAPPA,
    PIN_W,
    PRODUCTION_NUMERICS,
    RESULTS_DIR,
    access_by_seeding,
    attach_dispersion,
    breather_title_annotation,
    breathing_scan,
    cycle_averaged_spectrum,
    dispersive_wave_peaks,
    load_cavity_params,
    plot_breathing_scan,
    plot_optical_spectrum,
    plot_soliton_summary,
    write_breathing_csv,
)

# --- Committed artifact parameters (the audit record) -----------------------
DW_KAPPA = 8.0            # validated operating detuning [kappa]
SETTLE_N_TAU = 16384      # resolves both DW crossings (|mu| ~ 3000-3300)
SETTLE_RT = 12_000        # ~0.1 tau_th: seed fully relaxed onto the attractor
SETTLE_SEED = 0
CYCLE_AVG_RT = CYCLE_AVG_RT_8KAPPA   # 304 RT >= 2*T_b (T_b ~ 152-153 RT)

SCAN_DW_LO, SCAN_DW_HI, SCAN_DW_STEP = 7.0, 16.0, 0.5   # [kappa]
SCAN_RT = 4000
SCAN_N_TAU = 8192
SCAN_SEED = 0
# PRODUCTION_NUMERICS (shared with the settle run): n_substeps=4,
# dealias_two_thirds=True, edge_absorber=True, dispersion_validity_mask=False.

SPECTRUM_PNG = "dks_single_soliton_spectrum.png"
SPECTRUM_NPZ = "dks_single_soliton_spectrum.npz"   # raw cycle-averaged data
SUMMARY_PNG = "dks_single_soliton_summary.png"
SCAN_CSV = "dks_breathing_scan.csv"
SCAN_PNG = "dks_breathing_scan.png"
EXISTENCE_CSV = "dks_existence_map.csv"
PROVENANCE_JSON = "dks_artifact_provenance.json"


def merge_breathing_into_existence_csv(path: Path, scan_rows: list) -> None:
    """Add/update the V6 breathing columns of the existence-map CSV in place.

    The is_single (and every other pre-existing) column stands untouched; only
    the ``is_breather`` / ``breathing_period_rt`` / ``breathing_relstd``
    columns are (re)written, populated from the breathing scan for the
    detunings it covers and left empty elsewhere.
    """
    by_dw = {round(r["dw_over_kappa"], 3): r for r in scan_rows}
    with path.open(newline="") as f:
        reader = list(csv.reader(f))
    header, rows = reader[0], reader[1:]
    base_cols = [c for c in header if c not in
                 ("is_breather", "breathing_period_rt", "breathing_relstd")]
    keep = [header.index(c) for c in base_cols]
    out_header = base_cols + ["is_breather", "breathing_period_rt",
                              "breathing_relstd"]
    i_dw = base_cols.index("dw_over_kappa")
    out_rows = []
    for row in rows:
        base = [row[i] for i in keep]
        r = by_dw.get(round(float(base[i_dw]), 3))
        if r is None:
            base += ["", "", ""]
        else:
            base += [str(int(r["is_breather"])),
                     f"{r['breathing_period_rt']:.1f}",
                     f"{r['breathing_relstd']:.6f}"]
        out_rows.append(base)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(out_header)
        w.writerows(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-settle", action="store_true",
                    help="skip stages 1-3 (spectrum/summary PNGs)")
    ap.add_argument("--skip-scan", action="store_true",
                    help="skip stage 4 (breathing scan + existence-map merge)")
    ap.add_argument("--n-tau", type=int, default=SETTLE_N_TAU)
    ap.add_argument("--settle-rt", type=int, default=SETTLE_RT)
    ap.add_argument("--avg-rt", type=int, default=CYCLE_AVG_RT)
    ap.add_argument("--scan-rt", type=int, default=SCAN_RT)
    ap.add_argument("--scan-n-tau", type=int, default=SCAN_N_TAU)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Partial regenerations (--skip-*) must not drop the other stage's audit
    # record: start from the existing provenance and overwrite per stage.
    provenance = {}
    prov_path = RESULTS_DIR / PROVENANCE_JSON
    if prov_path.exists():
        with prov_path.open() as f:
            provenance = json.load(f)
    provenance.update({
        "script": "scripts/regenerate_dks_artifacts.py",
        "generated_unix_time": int(time.time()),
        "pin_w": PIN_W,
        "numerics": dict(PRODUCTION_NUMERICS),
        "settle": {"dw_kappa": DW_KAPPA, "n_tau": args.n_tau,
                   "t_slow_rt": args.settle_rt, "seed": SETTLE_SEED},
        "cycle_average": {"n_rt": args.avg_rt},
        "scan": {"dw_kappa_lo": SCAN_DW_LO, "dw_kappa_hi": SCAN_DW_HI,
                 "dw_kappa_step": SCAN_DW_STEP, "t_slow_rt": args.scan_rt,
                 "n_tau": args.scan_n_tau, "seed": SCAN_SEED},
    })

    if not args.skip_settle:
        print(f"[regen] stage 1 SETTLE: dw = {DW_KAPPA} kappa, "
              f"n_tau = {args.n_tau}, {args.settle_rt} RT, seed {SETTLE_SEED}, "
              f"numerics {PRODUCTION_NUMERICS}")
        cav = load_cavity_params()
        cav = attach_dispersion(cav, args.n_tau)
        dw = DW_KAPPA * cav.kappa
        res = access_by_seeding(dw, cav, t_slow=args.settle_rt,
                                seed=SETTLE_SEED, n_tau=args.n_tau, pin=PIN_W,
                                **PRODUCTION_NUMERICS)
        m = res["metrics"]
        assert res["is_single"], f"settle run is not a single soliton: {m}"

        print(f"[regen] stage 2 V6: stationarity = {m['stationarity']}, "
              f"T_b = {m['breathing_period_rt']:.1f} RT, "
              f"dU/U = {m['breathing_relstd']:.2%}")
        assert m["is_breather"], (
            "V6 no longer reports a breather at 8 kappa — the committed "
            "breather annotations would be stale. Investigate before "
            "regenerating."
        )
        title_extra = breather_title_annotation(m)

        print(f"[regen] stage 3 AVERAGE: {args.avg_rt} RT cycle average ...")
        sp = cycle_averaged_spectrum(res, cav, n_rt=args.avg_rt, pin=PIN_W,
                                     **PRODUCTION_NUMERICS)
        np.savez_compressed(
            RESULTS_DIR / SPECTRUM_NPZ, mu=sp["mu"],
            wavelength_nm=sp["wavelength_nm"], power_db=sp["power_db"],
            power_norm=sp["power_norm"], n_rt_averaged=sp["n_rt_averaged"],
            breathing_period_rt=m["breathing_period_rt"],
            breathing_relstd=m["breathing_relstd"],
        )
        plot_optical_spectrum(RESULTS_DIR / SPECTRUM_PNG, res["e_final"], cav,
                              dw, sp=sp, title_extra=title_extra)
        plot_soliton_summary(RESULTS_DIR / SUMMARY_PNG, res, cav, sp=sp,
                             title_extra=title_extra)

        dws = dispersive_wave_peaks(sp, dw)
        lam = np.asarray(sp["wavelength_nm"])
        db = np.asarray(sp["power_db"])
        ref = {f"db_at_{int(nm)}nm": float(db[int(np.argmin(np.abs(lam - nm)))])
               for nm in (1800.0, 2000.0)}
        # red-side sech-tail slope over 400 <= -mu <= 1500 (dB/mode)
        mu = np.asarray(sp["mu"])
        sel = (mu <= -400) & (mu >= -1500)
        slope = float(np.polyfit(np.abs(mu[sel]).astype(float), db[sel], 1)[0])
        provenance["measured"] = {
            "is_single": bool(res["is_single"]),
            "stationarity": m["stationarity"],
            "breathing_period_rt": m["breathing_period_rt"],
            "breathing_relstd": m["breathing_relstd"],
            "red_tail_slope_db_per_mode": slope,
            **ref,
            "dispersive_waves": [
                {"wavelength_nm": p["wavelength_nm"], "power_db": p["power_db"],
                 "prominence_db": p["prominence_db"], "mu": p["mu"]}
                for p in dws
            ],
        }
        print(f"[regen] reference levels: red tail slope {slope:.4f} dB/mode; "
              + "; ".join(f"{v:.1f} dB @ {k[6:-2]} nm" for k, v in ref.items())
              + "; DWs " + ", ".join(f"{p['power_db']:.1f} dB @ "
                                     f"{p['wavelength_nm']:.0f} nm" for p in dws))

    if not args.skip_scan:
        grid = np.round(np.arange(SCAN_DW_LO, SCAN_DW_HI + 1e-9, SCAN_DW_STEP), 3)
        print(f"[regen] stage 4 SCAN: dw = {grid[0]}..{grid[-1]} kappa "
              f"({grid.size} points), {args.scan_rt} RT, "
              f"n_tau = {args.scan_n_tau}, seed {SCAN_SEED}")
        cav_s = load_cavity_params()
        cav_s = attach_dispersion(cav_s, args.scan_n_tau)
        scan = breathing_scan(cav_s, grid, t_slow=args.scan_rt, seed=SCAN_SEED,
                              n_tau=args.scan_n_tau, pin=PIN_W,
                              **PRODUCTION_NUMERICS)
        write_breathing_csv(RESULTS_DIR / SCAN_CSV, scan)
        plot_breathing_scan(RESULTS_DIR / SCAN_PNG, scan, cav_s)
        merge_breathing_into_existence_csv(RESULTS_DIR / EXISTENCE_CSV,
                                           scan["rows"])
        provenance["scan_result"] = {
            "breathing_bands_kappa": scan["breathing_bands"],
            "stationary_windows_kappa": scan["stationary_windows"],
            "rows": [
                {"dw_over_kappa": r["dw_over_kappa"],
                 "is_single": bool(r["is_single"]),
                 "is_breather": bool(r["is_breather"]),
                 "is_stationary": bool(r["is_stationary"]),
                 "breathing_period_rt": r["breathing_period_rt"],
                 "breathing_relstd": r["breathing_relstd"]}
                for r in scan["rows"]
            ],
        }
        print(f"[regen] breathing sub-bands (kappa): {scan['breathing_bands']}")
        print(f"[regen] stationary (<0.1%) windows: {scan['stationary_windows']}")

    with (RESULTS_DIR / PROVENANCE_JSON).open("w") as f:
        json.dump(provenance, f, indent=2, default=float)
    print(f"[regen] provenance -> {RESULTS_DIR / PROVENANCE_JSON}")


if __name__ == "__main__":
    main()
