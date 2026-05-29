from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from logicpipe.types import ResourceProfile


@dataclass
class OfflineProfilerCalibration:
    layer_compute_ms: float
    layer_memory_mb: float
    activation_size_mb: float
    device_memory_mb: float
    inter_device_ms: float
    source: str = "heuristic"


class OfflineResourceProfiler:
    """
    Minimal offline profiler.

    This repo currently lacks the paper-grade measurement pipeline, so the profiler
    returns a deterministic heuristic profile that keeps the planner/runtime paths
    executable and explicitly labels the calibration source.
    """

    def _derive_calibration(self, config: Any, world_size: int) -> OfflineProfilerCalibration:
        hidden_size = float(getattr(config, "hidden_size", 4096))
        intermediate_size = float(getattr(config, "intermediate_size", hidden_size * 4))
        num_attention_heads = float(getattr(config, "num_attention_heads", 32))
        num_key_value_heads = float(getattr(config, "num_key_value_heads", num_attention_heads))
        bytes_per_elem = 2.0

        # Very rough per-layer weight footprint for LLaMA-style blocks.
        attn_params = 2.0 * hidden_size * hidden_size + 2.0 * hidden_size * hidden_size * (
            num_key_value_heads / max(num_attention_heads, 1.0)
        )
        mlp_params = 3.0 * hidden_size * intermediate_size
        norm_params = 2.0 * hidden_size
        layer_memory_mb = (attn_params + mlp_params + norm_params) * bytes_per_elem / (1024.0 ** 2)

        max_kv_cache_length = float(
            getattr(
                config,
                "max_kv_cache_length",
                getattr(config, "max_position_embeddings", 2048),
            )
        )
        head_dim = hidden_size / max(num_attention_heads, 1.0)
        activation_elems = 2.0 * num_key_value_heads * max_kv_cache_length * head_dim
        activation_size_mb = activation_elems * bytes_per_elem / (1024.0 ** 2)

        # Keep these heuristics conservative and monotonic with model size.
        layer_compute_ms = max(0.5, hidden_size / 2048.0)
        device_memory_mb = max(2048.0, layer_memory_mb * max(2.0, 32.0 / max(world_size, 1)))
        try:
            import torch  # local optional dependency during real runtime

            if torch.cuda.is_available():
                gpu_total_mb = float(torch.cuda.get_device_properties(0).total_memory) / (1024.0 ** 2)
                device_memory_mb = max(device_memory_mb, gpu_total_mb * 0.8)
        except Exception:
            pass
        inter_device_ms = max(0.05, activation_size_mb / 128.0)

        return OfflineProfilerCalibration(
            layer_compute_ms=layer_compute_ms,
            layer_memory_mb=layer_memory_mb,
            activation_size_mb=activation_size_mb,
            device_memory_mb=device_memory_mb,
            inter_device_ms=inter_device_ms,
            source="heuristic",
        )

    def profile(self, config: Any, world_size: int) -> ResourceProfile:
        num_layers = int(getattr(config, "num_hidden_layers"))
        calibration = self._derive_calibration(config=config, world_size=world_size)

        layer_compute_ms: List[float] = [calibration.layer_compute_ms for _ in range(num_layers)]
        layer_memory_mb: List[float] = [calibration.layer_memory_mb for _ in range(num_layers)]
        activation_size_mb: List[float] = [calibration.activation_size_mb for _ in range(num_layers)]
        device_slowdown: List[float] = [1.0 for _ in range(world_size)]
        device_memory_mb: List[float] = [calibration.device_memory_mb for _ in range(world_size)]
        inter_device_ms: List[List[float]] = []
        for src in range(world_size):
            row: List[float] = []
            for dst in range(world_size):
                row.append(0.0 if src == dst else calibration.inter_device_ms)
            inter_device_ms.append(row)

        return ResourceProfile(
            layer_compute_ms=layer_compute_ms,
            device_slowdown=device_slowdown,
            inter_device_ms=inter_device_ms,
            world_size=world_size,
            layer_memory_mb=layer_memory_mb,
            device_memory_mb=device_memory_mb,
            activation_size_mb=activation_size_mb,
            calibration_source=calibration.source,
        )
