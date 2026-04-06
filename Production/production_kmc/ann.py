from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .descriptor import DescriptorLayout


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "PyTorch is required for ANN inference. Install torch before running simulations."
        ) from exc
    return torch, nn


class MLPModel:
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int], output_dim: int):
        torch, nn = _require_torch()
        super().__init__()
        self.torch = torch
        self.nn = nn

        hidden_sizes = list(hidden_sizes)
        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")

        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def to(self, device: str):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()

    def load_state_dict(self, state_dict):
        # map from legacy names (hidden_layers.X / batch_norms.X / output_layer)
        # to sequential names if needed.
        if any(key.startswith("hidden_layers.") for key in state_dict.keys()):
            hidden_layer_ids = set()
            for key in state_dict.keys():
                if key.startswith("hidden_layers.") and key.count(".") >= 2:
                    try:
                        hidden_layer_ids.add(int(key.split(".")[1]))
                    except ValueError:
                        pass
            n_hidden = (max(hidden_layer_ids) + 1) if hidden_layer_ids else 0

            remapped = {}
            for key, value in state_dict.items():
                if key.startswith("hidden_layers."):
                    _, idx, suffix = key.split(".", 2)
                    layer_base = int(idx) * 3
                    remapped[f"{layer_base}.{suffix}"] = value
                elif key.startswith("batch_norms."):
                    _, idx, suffix = key.split(".", 2)
                    layer_base = int(idx) * 3 + 1
                    remapped[f"{layer_base}.{suffix}"] = value
                elif key.startswith("output_layer."):
                    # In the sequential form, each hidden block takes 3 slots
                    # (Linear, BatchNorm, ReLU), so output linear starts at n_hidden*3.
                    out_idx = n_hidden * 3
                    remapped[f"{out_idx}.{key.split('.', 1)[1]}"] = value
                else:
                    remapped[key] = value
            state_dict = remapped

        self.model.load_state_dict(state_dict)

    def __call__(self, x):
        return self.model(x)


@dataclass
class PredictionResult:
    barriers_eV: np.ndarray
    cache_hit: bool


class CachedBarrierPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        descriptor_layout: DescriptorLayout,
        device: str = "cpu",
        cache_size: int = 200_000,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.layout = descriptor_layout
        self.device = device
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.torch, _nn = _require_torch()
        self.model = self._load_model()

    def _load_model(self):
        torch = self.torch
        state = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        hidden_sizes = self.layout.hidden_sizes
        if not hidden_sizes:
            hidden_sizes = self._infer_hidden_sizes(state)

        mlp = MLPModel(
            input_dim=self.layout.input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=self.layout.output_dim,
        )
        mlp.load_state_dict(state)
        mlp.to(self.device)
        mlp.eval()
        return mlp

    @staticmethod
    def _infer_hidden_sizes(state_dict: Dict[str, np.ndarray]) -> List[int]:
        # Supports keys like hidden_layers.0.weight, hidden_layers.1.weight.
        sizes: Dict[int, int] = {}
        for key, tensor in state_dict.items():
            if key.startswith("hidden_layers.") and key.endswith(".weight"):
                idx = int(key.split(".")[1])
                sizes[idx] = int(tensor.shape[0])
        if sizes:
            return [sizes[i] for i in sorted(sizes)]

        # Fallback for sequential names: 0.weight, 3.weight, 6.weight, ...
        seq_linear = []
        for key, tensor in state_dict.items():
            if key.endswith(".weight") and key.split(".", 1)[0].isdigit() and len(tensor.shape) == 2:
                seq_linear.append((int(key.split(".", 1)[0]), int(tensor.shape[0]), int(tensor.shape[1])))
        if not seq_linear:
            raise ValueError("Could not infer hidden layer sizes from checkpoint state_dict")

        seq_linear.sort(key=lambda x: x[0])
        if len(seq_linear) < 2:
            raise ValueError("Unexpected checkpoint: not enough linear layers to infer architecture")
        return [row[1] for row in seq_linear[:-1]]

    @staticmethod
    def _descriptor_key(vector: np.ndarray) -> bytes:
        arr = np.asarray(vector, dtype=np.float32)
        return arr.tobytes()

    def _lookup_cache(self, key: bytes) -> np.ndarray | None:
        if self.cache_size == 0:
            return None
        hit = self._cache.get(key)
        if hit is None:
            return None
        self._cache.move_to_end(key)
        return hit

    def _store_cache(self, key: bytes, value: np.ndarray) -> None:
        if self.cache_size == 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def predict(self, descriptor_vector: np.ndarray) -> PredictionResult:
        descriptor_vector = np.asarray(descriptor_vector, dtype=np.float32)
        if descriptor_vector.shape != (self.layout.input_dim,):
            raise ValueError(
                f"Descriptor vector shape mismatch. Expected ({self.layout.input_dim},), got {descriptor_vector.shape}"
            )

        key = self._descriptor_key(descriptor_vector)
        cached = self._lookup_cache(key)
        if cached is not None:
            return PredictionResult(barriers_eV=cached.copy(), cache_hit=True)

        torch = self.torch
        x = torch.from_numpy(descriptor_vector[None, :]).to(self.device)
        with torch.no_grad():
            y = self.model(x)
            pred_norm = y.detach().cpu().numpy().reshape(-1)

        barriers = pred_norm * self.layout.target_max_values
        barriers = barriers.astype(np.float64, copy=False)
        self._store_cache(key, barriers)
        return PredictionResult(barriers_eV=barriers.copy(), cache_hit=False)
