"""Batch simulation dataset generation module.

This module orchestrates parameter sweeps and repeated simulations, then
serializes trajectories and labels for downstream model training.
"""

import itertools
import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from simulator.lle_solver import (
    _PER_TRAJ,
    _STATE_LABELER,
    _load_config,
    _thermal_params,
    d2_to_beta2_lle,
    gamma_nlse_to_lle,
)
from simulator.noise_models import TotalNoise, _load_config as nm_load_cfg


class DatasetGenerator:
    """Generate and store synthetic LLE trajectories for training."""

    SEGMENT_RT = 500
    HOLD_RT = 200

    def __init__(
        self,
        param_grid: dict[str, list],
        config_path: str | Path | None = None,
        output_dir: str | Path = "data/synthetic",
        batch_size: int = 64,
        n_tau: int = 512,
        snapshot_interval: int = 100,
        seed: int = 42,
    ):
        self.param_grid = param_grid
        self.config_path = str(config_path) if config_path is not None else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = int(batch_size)
        self.n_tau = int(n_tau)
        self.snapshot_interval = int(snapshot_interval)
        self.base_key = jax.random.PRNGKey(seed)

        self.config = _load_config(self.config_path)
        self.fsr_hz = float(self.config["fsr_hz"])
        self.t_r = 1.0 / self.fsr_hz
        self.kappa_i = float(self.config["kappa_i_rad_per_s"])
        self.kappa_c = self.kappa_i
        self.kappa = self.kappa_i + self.kappa_c

        if "gamma_LLE_per_J_per_s" in self.config:
            self.gamma = float(self.config["gamma_LLE_per_J_per_s"])
        else:
            self.gamma = float(
                gamma_nlse_to_lle(self.config["gamma_per_W_m"], self.fsr_hz)
            )

        self.beta2 = float(d2_to_beta2_lle(self.config["d2_rad_per_s2"], self.fsr_hz))
        self.beta = [self.beta2]

        self.p_th = (self.kappa / 2.0) ** 2 / (self.gamma * self.t_r * self.kappa_c)

        self.full_simulation_list = [
            {
                "pin": float(pin),
                "sweep_rate": float(sweep_rate),
                "Gamma_th": float(gamma_th),
                "noise_scale": float(noise_scale),
            }
            for pin, sweep_rate, gamma_th, noise_scale in itertools.product(
                self.param_grid["pin"],
                self.param_grid["sweep_rate"],
                self.param_grid["Gamma_th"],
                self.param_grid["noise_scale"],
            )
        ]

    def _make_keys(self, batch_global_idx: int, B: int) -> tuple[jax.Array, jax.Array]:
        batch_key = jax.random.fold_in(self.base_key, int(batch_global_idx))
        field_key, noise_key = jax.random.split(batch_key)
        key_arr = jax.random.split(field_key, B)
        noise_keys = jax.random.split(noise_key, B)
        return key_arr, noise_keys

    def _forward_fill_labels(self, label_history: np.ndarray, t_total: int) -> np.ndarray:
        # Each snapshot label is forward-filled for snapshot_interval round trips.
        # At segment boundaries the last snapshot of a segment may cover fewer than
        # snapshot_interval round trips, introducing a label lag of up to
        # (snapshot_interval - 1) steps. Acceptable for training data.
        labels = np.repeat(label_history, self.snapshot_interval, axis=1)
        return labels[:, :t_total].astype(np.int32)

    def simulate_batch(self, params: list[dict], batch_global_idx: int) -> dict:
        B = len(params)
        if B == 0:
            raise ValueError("simulate_batch received empty params")

        sweep_rates = {float(p["sweep_rate"]) for p in params}
        if len(sweep_rates) != 1:
            raise ValueError("All batch params must share the same sweep_rate")

        sweep_rate = sweep_rates.pop()
        pins = {float(p["pin"]) for p in params}
        if len(pins) != 1:
            raise ValueError("All batch params must share the same pin")
        pin_scalar = pins.pop()

        gamma_ths = {float(p["Gamma_th"]) for p in params}
        if len(gamma_ths) != 1:
            raise ValueError("All batch params must share the same Gamma_th")
        gamma_th_scalar = gamma_ths.pop()

        noise_scale_arr = jnp.array([float(p["noise_scale"]) for p in params], dtype=jnp.float32)
        n_sweep_segments = int(math.ceil(8.0 * self.kappa / (sweep_rate * self.SEGMENT_RT)))

        key_arr, noise_keys = self._make_keys(batch_global_idx=batch_global_idx, B=B)

        thermal = _thermal_params(self.config_path)
        thermal["Gamma_th"] = gamma_th_scalar
        thermal["kappa_i"] = self.kappa_i
        thermal = {k: jnp.array(v, dtype=jnp.float32) for k, v in thermal.items()}

        noise_model = TotalNoise(nm_load_cfg(self.config_path))

        outputs = {
            "P_trans": [],
            "U_int": [],
            "DeltaT": [],
            "delta_omega_eff": [],
            "label_history": [],
            "E_snapshots": [],
        }
        # Cold-start sentinels: all-zeros triggers CW+noise / zero thermal initialisation
        e_carry = jnp.zeros((B, self.n_tau), dtype=jnp.complex64)
        delta_t_carry = jnp.zeros((B,), dtype=jnp.float32)

        total_segments = n_sweep_segments + 1
        for seg_idx in range(total_segments):
            if seg_idx < n_sweep_segments:
                delta_center = (3.0 * self.kappa) - sweep_rate * (
                    seg_idx * self.SEGMENT_RT + self.SEGMENT_RT / 2.0
                )
                t_seg = self.SEGMENT_RT
            else:
                delta_center = -5.0 * self.kappa
                t_seg = self.HOLD_RT

            delta_arr = jnp.full((B,), delta_center, dtype=jnp.float32)
            seg_field_keys = jax.random.fold_in(key_arr, seg_idx)
            seg_noise_keys = jax.random.fold_in(noise_keys, seg_idx)
            noise_seqs = jax.vmap(
                lambda k, scale: noise_model.sample(k, t_seg) * scale,
                in_axes=(0, 0),
            )(seg_noise_keys, noise_scale_arr)

            out = _PER_TRAJ(
                delta_arr,
                float(pin_scalar),
                int(t_seg),
                tuple(self.beta),
                float(self.gamma),
                float(self.kappa),
                float(self.kappa_c),
                int(self.n_tau),
                float(self.t_r),
                1.0,
                int(self.snapshot_interval),
                seg_field_keys,
                thermal,
                _STATE_LABELER,
                noise_seqs,
                e_carry,
                delta_t_carry,
            )
            # Carry field and thermal state forward across segment boundary
            e_carry = jnp.array(out["E_snapshots"][:, -1, :], dtype=jnp.complex64)
            delta_t_carry = jnp.array(out["delta_t_final"], dtype=jnp.float32)

            outputs["P_trans"].append(np.asarray(out["P_trans_history"], dtype=np.float32))
            outputs["U_int"].append(np.asarray(out["U_int_history"], dtype=np.float32))
            outputs["DeltaT"].append(np.asarray(out["DeltaT_history"], dtype=np.float32))
            outputs["delta_omega_eff"].append(
                np.asarray(out["delta_omega_eff_history"], dtype=np.float32)
            )
            outputs["label_history"].append(np.asarray(out["label_history"], dtype=np.int32))
            outputs["E_snapshots"].append(np.asarray(out["E_snapshots"], dtype=np.complex64))

        p_trans = np.concatenate(outputs["P_trans"], axis=1)
        u_int = np.concatenate(outputs["U_int"], axis=1)
        delta_t = np.concatenate(outputs["DeltaT"], axis=1)
        delta_eff = np.concatenate(outputs["delta_omega_eff"], axis=1)
        label_history = np.concatenate(outputs["label_history"], axis=1)
        e_snapshots = np.concatenate(outputs["E_snapshots"], axis=1)

        t_total = p_trans.shape[1]
        labels = self._forward_fill_labels(label_history, t_total)

        return {
            "P_trans": p_trans,
            "U_int": u_int,
            "DeltaT": delta_t,
            "delta_omega_eff": delta_eff,
            "labels": labels,
            "E_snapshots": e_snapshots,
        }

    def save_batch(self, results: dict, params: list[dict], start_idx: int, h5file) -> None:
        B = len(params)
        for i in range(B):
            sim_name = f"sim_{start_idx + i}"
            if sim_name in h5file:
                del h5file[sim_name]
            grp = h5file.create_group(sim_name)

            grp.create_dataset(
                "P_trans", data=results["P_trans"][i], compression="gzip", compression_opts=4
            )
            grp.create_dataset(
                "U_int", data=results["U_int"][i], compression="gzip", compression_opts=4
            )
            grp.create_dataset(
                "DeltaT", data=results["DeltaT"][i], compression="gzip", compression_opts=4
            )
            grp.create_dataset(
                "delta_omega_eff",
                data=results["delta_omega_eff"][i],
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset(
                "labels", data=results["labels"][i], compression="gzip", compression_opts=4
            )
            grp.create_dataset(
                "E_snapshots",
                data=results["E_snapshots"][i],
                compression="gzip",
                compression_opts=4,
            )

            grp.attrs["pin"] = float(params[i]["pin"])
            grp.attrs["sweep_rate"] = float(params[i]["sweep_rate"])
            grp.attrs["Gamma_th"] = float(params[i]["Gamma_th"])
            grp.attrs["noise_scale"] = float(params[i]["noise_scale"])

    def generate_full_dataset(self, n_total: int = 45_000) -> None:
        param_list = list(self.full_simulation_list)
        repeats = int(math.ceil(n_total / max(1, len(param_list))))
        expanded = (param_list * repeats)[:n_total]
        expanded.sort(key=lambda x: (x["sweep_rate"], x["pin"], x["Gamma_th"]))

        out_path = self.output_dir / "dataset.h5"
        with h5py.File(out_path, "a") as h5file:
            metadata = h5file.require_group("metadata")
            metadata.attrs["param_grid_json"] = json.dumps(self.param_grid)
            metadata.attrs["generation_date"] = datetime.now().isoformat()
            metadata.attrs["n_tau"] = self.n_tau
            metadata.attrs["snapshot_interval"] = self.snapshot_interval
            metadata.attrs["p_th_watts"] = float(self.p_th)
            metadata.attrs[
                "class_scheme"
            ] = "0=off,1=CW,2=MI,3=chaotic,4=multi,5=crystal,6=single"

            resume_idx = 0
            checkpoint = h5file.require_group("checkpoint")
            if "last_completed_idx" in checkpoint.attrs:
                resume_idx = int(checkpoint.attrs["last_completed_idx"]) + 1

            grouped_batches = []
            idx = resume_idx
            while idx < len(expanded):
                sweep   = expanded[idx]["sweep_rate"]
                pin_val = expanded[idx]["pin"]
                gth_val = expanded[idx]["Gamma_th"]
                chunk = []
                while (
                    idx < len(expanded)
                    and len(chunk) < self.batch_size
                    and expanded[idx]["sweep_rate"] == sweep
                    and expanded[idx]["pin"]       == pin_val
                    and expanded[idx]["Gamma_th"]  == gth_val
                ):
                    chunk.append(expanded[idx])
                    idx += 1
                grouped_batches.append((idx - len(chunk), chunk))

            completed = resume_idx
            pbar = tqdm(
                total=n_total,
                desc="Generating dataset",
                unit="sim",
                initial=resume_idx,
            )

            try:
                for batch_idx, (start_idx, batch_params) in enumerate(grouped_batches):
                    if not batch_params:
                        continue
                    results = self.simulate_batch(
                        batch_params,
                        batch_global_idx=(start_idx // max(1, self.batch_size)) + batch_idx,
                    )
                    self.save_batch(results, batch_params, start_idx=start_idx, h5file=h5file)

                    completed += len(batch_params)
                    pbar.update(len(batch_params))

                    if completed % 1000 == 0:
                        checkpoint.attrs["last_completed_idx"] = completed - 1
                        h5file.flush()
            except KeyboardInterrupt:
                checkpoint.attrs["last_completed_idx"] = max(completed - 1, -1)
                h5file.flush()
                print("Dataset generation interrupted. Progress saved.")
            finally:
                pbar.close()

            checkpoint.attrs["last_completed_idx"] = max(completed - 1, -1)
            h5file.flush()
            self.print_summary(h5file)

    def print_summary(self, h5file) -> None:
        sim_keys = sorted(k for k in h5file.keys() if k.startswith("sim_"))
        if not sim_keys:
            print("No simulations found in dataset.")
            return

        labels_all = []
        traj_lengths = []
        for key in sim_keys:
            labels = np.asarray(h5file[key]["labels"][:], dtype=np.int32)
            labels_all.append(labels)
            traj_lengths.append(labels.shape[0])

        flat_labels = np.concatenate(labels_all, axis=0)
        counts = np.bincount(flat_labels, minlength=7)
        total_labels = max(1, int(flat_labels.shape[0]))

        file_size_mb = Path(h5file.filename).stat().st_size / (1024 * 1024)
        mean_len = float(np.mean(traj_lengths)) if traj_lengths else 0.0

        print(f"Total trajectories : {len(sim_keys)}")
        print(f"Total round trips  : {sum(traj_lengths)}")
        print(f"File size          : {file_size_mb:.1f} MB")
        print(f"Mean traj length   : {mean_len:.0f} round trips")
        print("Class distribution :")
        names = [
            "off",
            "CW",
            "MI",
            "chaotic",
            "multi-soliton",
            "soliton-crystal",
            "single-soliton",
        ]
        for i, name in enumerate(names):
            pct = 100.0 * counts[i] / total_labels
            print(f"  [{i}] {name:<16}: {pct:5.2f}%")


def generate_full_dataset(
    param_grid: dict[str, list] | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path = "data/synthetic",
    n_total: int = 45_000,
    batch_size: int = 64,
    seed: int = 42,
) -> None:
    if param_grid is None:
        param_grid = {
            "pin": [0.7, 1.0, 1.5, 2.0, 3.0],
            # FIXED: rates in rad/s per round trip, matching 0.003–0.3 GHz/µs physical range.
            # Maximum safe rate: step_size = sweep_rate × SEGMENT_RT < 0.5 κ.
            # With κ ≈ 1.215e8 and SEGMENT_RT = 500: max ≈ 1.2e5.
            "sweep_rate": [3e2, 1e3, 5e3, 1e4, 5e4, 1e5],
            "Gamma_th": [0.05, 0.1, 0.2],
            "noise_scale": [0.5, 1.0, 2.0],
        }

    generator = DatasetGenerator(
        param_grid=param_grid,
        config_path=config_path,
        output_dir=output_dir,
        batch_size=batch_size,
        seed=seed,
    )
    generator.generate_full_dataset(n_total=n_total)
