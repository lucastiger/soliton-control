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
1. SETTLE   Seeded single-DKS run at DW_KAPPA = OPERATING_DW_KAPPA, pin = 0.214 W,
            n_tau = 16384, 12000 round trips, seed 0, production numerics
            (float64, n_substeps = 4, 2/3 dealias ON, edge absorber ON,
            dispersion-validity mask OFF).
2. V6       Stationarity check on the U_int history. At the production operating
            point (OPERATING_DW_KAPPA) the attractor is a STATIONARY single DKS,
            so the script aborts if V6 reports a breather or dU/U >=
            STATIONARY_RELSTD (0.1%) — the committed spectrum artifacts require a
            settled stationary state (if intentionally running the breather
            point, use cycle_averaged_spectrum instead).
3. SNAPSHOT Stationary single-snapshot spectrum: continue the settled trajectory
            for CHECK_RT round trips at snapshot_interval = 1 and take the FINAL
            snapshot's spectrum (phase-independent for a stationary attractor).
            The continuation also yields the stability diagnostics
            (energy_rel_std, spectrum_max_dev_db, V6 is_breather, centroid
            drift); the script aborts unless energy_rel_std < STATIONARY_RELSTD
            and V6 is non-breathing. Regenerates dks_single_soliton_spectrum.png
            and dks_single_soliton_summary.png with a "single snapshot" label.
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
    OPERATING_DW_KAPPA,
    PIN_W,
    PRODUCTION_NUMERICS,
    RESULTS_DIR,
    STATIONARY_RELSTD,
    access_by_seeding,
    attach_dispersion,
    breathing_scan,
    dispersive_wave_peaks,
    load_cavity_params,
    plot_breathing_scan,
    plot_optical_spectrum,
    plot_soliton_summary,
    stationary_snapshot_spectrum,
    write_breathing_csv,
)

# --- Committed artifact parameters (the audit record) -----------------------
DW_KAPPA = OPERATING_DW_KAPPA   # production operating detuning [kappa] (authoritative)
SETTLE_N_TAU = 16384      # resolves both DW crossings (|mu| ~ 3000-3300)
SETTLE_RT = 12_000        # ~0.1 tau_th: seed fully relaxed onto the attractor
SETTLE_SEED = 0
# Stage-3 stationarity-check continuation length. 304 RT (>> the 200 RT V6 needs
# for a reliable breather/stationary readout) is ample for a settled DKS; reuses
# the validated CYCLE_AVG_RT_8KAPPA number of round trips.
CHECK_RT = CYCLE_AVG_RT_8KAPPA

SCAN_DW_LO, SCAN_DW_HI, SCAN_DW_STEP = 7.0, 16.0, 0.5   # [kappa]
SCAN_RT = 4000
SCAN_N_TAU = 8192
SCAN_SEED = 0
# PRODUCTION_NUMERICS (shared with the settle run): n_substeps=4,
# dealias_two_thirds=True, edge_absorber=True, dispersion_validity_mask=False.

SPECTRUM_PNG = "dks_single_soliton_spectrum.png"
SPECTRUM_NPZ = "dks_single_soliton_spectrum.npz"   # raw single-snapshot data
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
    ap.add_argument("--check-rt", dest="check_rt", type=int, default=CHECK_RT,
                    help="round trips for the stage-3 stationarity check")
    # Back-compat hidden alias for --check-rt (formerly the cycle-average length).
    ap.add_argument("--avg-rt", dest="check_rt", type=int,
                    default=argparse.SUPPRESS, help=argparse.SUPPRESS)
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
        "scan": {"dw_kappa_lo": SCAN_DW_LO, "dw_kappa_hi": SCAN_DW_HI,
                 "dw_kappa_step": SCAN_DW_STEP, "t_slow_rt": args.scan_rt,
                 "n_tau": args.scan_n_tau, "seed": SCAN_SEED},
    })
    # The forced-breather cycle-average block was renamed to "stationarity_check";
    # drop the stale key when updating a provenance file from the old path.
    provenance.pop("cycle_average", None)

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
        assert not m["is_breather"] and m["breathing_relstd"] < STATIONARY_RELSTD, (
            f"expected a stationary DKS at {DW_KAPPA}κ but V6 reports "
            f"is_breather={m['is_breather']}, dU/U={m['breathing_relstd']:.3%} "
            f">= {STATIONARY_RELSTD:.1%}. If intentionally running the breather "
            f"point, use cycle_averaged_spectrum instead.")
        title_extra = ""   # stationary DKS: no breather annotation

        print(f"[regen] stage 3 SNAPSHOT: stationary single snapshot, "
              f"{args.check_rt} RT stationarity check ...")
        sp = stationary_snapshot_spectrum(res, cav, n_check_rt=args.check_rt,
                                          pin=PIN_W, **PRODUCTION_NUMERICS)
        assert sp["energy_rel_std"] < STATIONARY_RELSTD, (
            f"stage-3 continuation is not stationary: energy_rel_std = "
            f"{sp['energy_rel_std']:.3%} >= {STATIONARY_RELSTD:.1%} "
            f"(spectrum_max_dev_db = {sp['spectrum_max_dev_db']:.2f} dB).")
        assert not sp["is_breather"], (
            f"stage-3 continuation V6 reports a breather (dU/U = "
            f"{sp['breathing_relstd']:.3%}); expected a stationary DKS.")
        print(f"[regen] stationarity: energy_rel_std = {sp['energy_rel_std']:.4%}, "
              f"spectrum_max_dev = {sp['spectrum_max_dev_db']:.2f} dB, "
              f"centroid_drift = {sp['centroid_drift_modes']:.3f} modes")
        np.savez_compressed(
            RESULTS_DIR / SPECTRUM_NPZ, mu=sp["mu"],
            wavelength_nm=sp["wavelength_nm"], power_db=sp["power_db"],
            power_norm=sp["power_norm"], n_rt_checked=sp["n_rt_checked"],
            energy_rel_std=sp["energy_rel_std"],
            spectrum_max_dev_db=sp["spectrum_max_dev_db"],
        )
        plot_optical_spectrum(RESULTS_DIR / SPECTRUM_PNG, res["e_final"], cav,
                              dw, sp=sp, title_extra=title_extra)
        plot_soliton_summary(RESULTS_DIR / SUMMARY_PNG, res, cav, sp=sp,
                             title_extra=title_extra)
        provenance["stationarity_check"] = {
            "n_rt": args.check_rt,
            "energy_rel_std": sp["energy_rel_std"],
            "spectrum_max_dev_db": sp["spectrum_max_dev_db"],
        }

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
            "stationarity": "stationary",
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
