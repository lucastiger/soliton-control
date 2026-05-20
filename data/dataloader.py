"""PyTorch-compatible data loading utilities for synthetic TFLN soliton data."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

STATE_NAMES: dict[int, str] = {
    0: "off",
    1: "CW",
    2: "MI",
    3: "chaotic",
    4: "multi_soliton",
    5: "soliton_crystal",
    6: "single_soliton",
}
N_CLASSES: int = 7


def _get_split_indices(n_traj: int, random_state: int = 42) -> dict[str, list[int]]:
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_traj).tolist()
    n_val = max(1, round(0.10 * n_traj))
    n_test = max(1, round(0.10 * n_traj))
    n_train = n_traj - n_val - n_test
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


class SolitonDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path = "data/synthetic/dataset.h5",
        split: str = "train",
        W: int = 200,
        H: int = 50,
        stride: int = 10,
        kappa: float = 1.214e9,
        Q_i: float = 2e6,
        FSR: float = 200e9,
        preload: bool = False,
        max_ram_gb: float = 16.0,
        random_state: int = 42,
    ) -> None:
        self._h5_path = Path(h5_path)
        self.split = split
        self.W = W
        self.H = H
        self.stride = stride
        self.kappa = float(kappa)
        self.Q_i = float(Q_i)
        self.FSR = float(FSR)
        self.preload = preload
        self.max_ram_gb = float(max_ram_gb)
        self.random_state = random_state

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}.")
        if self.W <= 0 or self.H <= 0 or self.stride <= 0:
            raise ValueError("W, H, and stride must be positive integers.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")

        self._h5file: h5py.File | None = None
        self._data: dict[str, list[np.ndarray]] | None = None

        with h5py.File(self._h5_path, "r") as h5file:
            traj_keys = sorted(
                [k for k in h5file.keys() if k.startswith("sim_")],
                key=lambda k: int(k.split("_")[1]),
            )
            self._traj_keys = traj_keys
            n_traj = len(traj_keys)
            split_indices = _get_split_indices(n_traj=n_traj, random_state=self.random_state)
            self._local_to_global = split_indices[self.split]

            split_keys = [traj_keys[gidx] for gidx in self._local_to_global]
            n_split_traj = len(split_keys)

            self._P0 = np.empty(n_split_traj, dtype=np.float32)
            self._kappa_arr = np.full(n_split_traj, self.kappa, dtype=np.float32)
            self._traj_lengths = np.empty(n_split_traj, dtype=np.int64)
            pins = np.empty(n_split_traj, dtype=np.float32)
            sweep_rates = np.empty(n_split_traj, dtype=np.float32)

            index_list: list[tuple[int, int]] = []
            labels_list: list[np.ndarray] = []

            for local_idx, key in enumerate(split_keys):
                grp = h5file[key]
                T = grp["P_trans"].shape[0]
                labels = grp["labels"][:]
                self._traj_lengths[local_idx] = T
    
                p0 = float(grp["P_trans"][labels == 1].mean())
                if p0 < 1e-12:
                    p0 = float(grp["P_trans"][:].mean())
                if p0 < 1e-12:
                    p0 = 1.0
                self._P0[local_idx] = np.float32(p0)

                pins[local_idx] = np.float32(float(grp.attrs["pin"]))
                sweep_rates[local_idx] = np.float32(float(grp.attrs["sweep_rate"]))

                max_start = T - self.W - self.H
                if max_start >= 0:
                    starts = list(range(0, max_start + 1, self.stride))
                    index_list.extend((local_idx, t_start) for t_start in starts)
                    labels_list.append(labels[self.W - 1 : T - self.H : self.stride].astype(np.int32, copy=False))

            self._index = index_list
            if labels_list:
                self._window_labels = np.concatenate(labels_list, axis=0).astype(np.int32, copy=False)
            else:
                self._window_labels = np.empty((0,), dtype=np.int32)

            self._context = np.stack(
                [
                    np.log10(np.maximum(pins, 1e-30) / 1e-3, dtype=np.float32),
                    np.full(n_split_traj, np.float32(math.log10(self.Q_i / 1e6)), dtype=np.float32),
                    np.full(n_split_traj, np.float32(self.FSR / 200e9), dtype=np.float32),
                    np.log10(np.maximum(sweep_rates, 1e-30) / 1e3, dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32, copy=False)

            if self.preload:
                total_bytes = int(self._traj_lengths.sum()) * 5 * 4
                if total_bytes > self.max_ram_gb * 1e9:
                    raise RuntimeError(
                        f"Preload would require {total_bytes/1e9:.1f} GB, exceeding limit of {self.max_ram_gb} GB."
                    )

                self._data = {
                    "P_trans": [],
                    "delta_omega_eff": [],
                    "labels": [],
                    "U_int": [],
                    "DeltaT": [],
                }
                for key in split_keys:
                    grp = h5file[key]
                    self._data["P_trans"].append(grp["P_trans"][:])
                    self._data["delta_omega_eff"].append(grp["delta_omega_eff"][:])
                    self._data["labels"].append(grp["labels"][:])
                    self._data["U_int"].append(grp["U_int"][:])
                    self._data["DeltaT"].append(grp["DeltaT"][:])

        self._h5file = None

    def __len__(self) -> int:
        return len(self._index)

    @functools.cached_property
    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self._window_labels, minlength=N_CLASSES)
        total = len(self._index)
        weights = [total / (N_CLASSES * max(int(counts[c]), 1)) for c in range(N_CLASSES)]
        return torch.tensor(weights, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        local_traj_idx, t_start = self._index[idx]
        t_mid = t_start + self.W
        t_end = t_mid + self.H

        if self._data is not None:
            p_trans = self._data["P_trans"][local_traj_idx]
            delta_omega_eff = self._data["delta_omega_eff"][local_traj_idx]
            labels = self._data["labels"][local_traj_idx]
            u_int = self._data["U_int"][local_traj_idx]
            delta_t = self._data["DeltaT"][local_traj_idx]
        else:
            if self._h5file is None:
                self._h5file = h5py.File(self._h5_path, "r")
            key = self._traj_keys[self._local_to_global[local_traj_idx]]
            grp = self._h5file[key]
            p_trans = grp["P_trans"]
            delta_omega_eff = grp["delta_omega_eff"]
            labels = grp["labels"]
            u_int = grp["U_int"]
            delta_t = grp["DeltaT"]

        p0 = float(self._P0[local_traj_idx])

        window = np.asarray(p_trans[t_start:t_mid], dtype=np.float32) / p0
        future_detuning = np.asarray(delta_omega_eff[t_mid:t_end], dtype=np.float32) / self._kappa_arr[local_traj_idx]
        future_p_trans = np.asarray(p_trans[t_mid:t_end], dtype=np.float32) / p0
        future_u_int = np.asarray(u_int[t_mid:t_end], dtype=np.float32)
        future_delta_t = np.asarray(delta_t[t_mid:t_end], dtype=np.float32)
        label_value = int(labels[t_mid - 1])

        return {
            "x": torch.tensor(window, dtype=torch.float32).unsqueeze(-1),
            "context": torch.tensor(self._context[local_traj_idx], dtype=torch.float32),
            "label": torch.tensor(label_value, dtype=torch.int64),
            "future_detuning": torch.tensor(future_detuning, dtype=torch.float32),
            "future_P_trans": torch.tensor(future_p_trans, dtype=torch.float32),
            "future_U_int": torch.tensor(future_u_int, dtype=torch.float32),
            "future_DeltaT": torch.tensor(future_delta_t, dtype=torch.float32),
        }

    def describe(self) -> None:
        n_traj = len(self._local_to_global)
        n_windows = len(self._index)
        counts = np.bincount(self._window_labels, minlength=N_CLASSES)
        pct = (counts / max(n_windows, 1)) * 100.0

        print(f"Split: {self.split}")
        print(f"Trajectories: {n_traj}")
        print(f"Total windows: {n_windows}")
        print("Class distribution:")
        for c in range(N_CLASSES):
            print(f"  {c} ({STATE_NAMES[c]}): {counts[c]} ({pct[c]:.2f}%)")
        print(f"class_weights: {self.class_weights}")
        print(
            f"P0 stats: min={self._P0.min():.6g}, max={self._P0.max():.6g}, "
            f"mean={self._P0.mean():.6g}"
        )
        print(
            f"Trajectory length stats: min={self._traj_lengths.min()}, "
            f"max={self._traj_lengths.max()}, mean={self._traj_lengths.mean():.2f}"
        )


def get_dataloaders(
    h5_path: str | Path = "data/synthetic/dataset.h5",
    W: int = 200,
    H: int = 50,
    stride: int = 10,
    kappa: float = 1.214e9,
    Q_i: float = 2e6,
    FSR: float = 200e9,
    batch_size: int = 512,
    num_workers: int = 4,
    preload: bool = False,
    max_ram_gb: float = 16.0,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader)."""
    common_kwargs: dict[str, Any] = {
        "h5_path": h5_path,
        "W": W,
        "H": H,
        "stride": stride,
        "kappa": kappa,
        "Q_i": Q_i,
        "FSR": FSR,
        "preload": preload,
        "max_ram_gb": max_ram_gb,
        "random_state": random_state,
    }

    train_ds = SolitonDataset(split="train", **common_kwargs)
    val_ds = SolitonDataset(split="val", **common_kwargs)
    test_ds = SolitonDataset(split="test", **common_kwargs)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
