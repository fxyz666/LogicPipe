from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class ResourceProfile:
    """Offline hardware profile used by partition planning."""

    layer_compute_ms: List[float]
    device_slowdown: List[float]
    inter_device_ms: List[List[float]]
    world_size: int
    layer_memory_mb: List[float] = field(default_factory=list)
    device_memory_mb: List[float] = field(default_factory=list)
    activation_size_mb: List[float] = field(default_factory=list)
    calibration_source: str = "heuristic"

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer_compute_ms": self.layer_compute_ms,
            "device_slowdown": self.device_slowdown,
            "inter_device_ms": self.inter_device_ms,
            "world_size": self.world_size,
            "layer_memory_mb": self.layer_memory_mb,
            "device_memory_mb": self.device_memory_mb,
            "activation_size_mb": self.activation_size_mb,
            "calibration_source": self.calibration_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceProfile":
        layer_compute_ms = [float(v) for v in payload["layer_compute_ms"]]
        device_slowdown = [float(v) for v in payload["device_slowdown"]]
        inter_device_ms = [
            [float(v) for v in row] for row in payload["inter_device_ms"]
        ]
        return cls(
            layer_compute_ms=layer_compute_ms,
            device_slowdown=device_slowdown,
            inter_device_ms=inter_device_ms,
            world_size=int(payload["world_size"]),
            layer_memory_mb=[float(v) for v in payload.get("layer_memory_mb", [])],
            device_memory_mb=[float(v) for v in payload.get("device_memory_mb", [])],
            activation_size_mb=[float(v) for v in payload.get("activation_size_mb", [])],
            calibration_source=str(payload.get("calibration_source", "heuristic")),
        )


@dataclass
class PartitionPlan:
    """DP output for contiguous layer partitioning."""

    stage_num_hidden_layers_list: List[int]
    bottleneck_ms: float
    selected_devices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage_num_hidden_layers_list": self.stage_num_hidden_layers_list,
            "bottleneck_ms": self.bottleneck_ms,
            "selected_devices": self.selected_devices,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PartitionPlan":
        stage_num_hidden_layers_list = [
            int(v) for v in payload["stage_num_hidden_layers_list"]
        ]
        selected_devices = [int(v) for v in payload.get("selected_devices", [])]
        return cls(
            stage_num_hidden_layers_list=stage_num_hidden_layers_list,
            bottleneck_ms=float(payload["bottleneck_ms"]),
            selected_devices=selected_devices,
        )


@dataclass
class OfflineArtifactMetadata:
    """Metadata persisted with offline artifacts to verify compatibility."""

    config_class: str
    config_name_or_path: Optional[str]
    model_type: Optional[str]
    world_size: int
    num_hidden_layers: int
    num_stages: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "config_class": self.config_class,
            "config_name_or_path": self.config_name_or_path,
            "model_type": self.model_type,
            "world_size": self.world_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_stages": self.num_stages,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OfflineArtifactMetadata":
        return cls(
            config_class=str(payload["config_class"]),
            config_name_or_path=payload.get("config_name_or_path"),
            model_type=payload.get("model_type"),
            world_size=int(payload["world_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_stages=int(payload["num_stages"]),
        )

    @classmethod
    def from_config(
        cls,
        config: Any,
        world_size: int,
        num_hidden_layers: int,
        num_stages: int,
    ) -> "OfflineArtifactMetadata":
        return cls(
            config_class=type(config).__name__,
            config_name_or_path=getattr(config, "name_or_path", None),
            model_type=getattr(config, "model_type", None),
            world_size=world_size,
            num_hidden_layers=num_hidden_layers,
            num_stages=num_stages,
        )

    def assert_compatible_with(self, other: "OfflineArtifactMetadata") -> None:
        mismatches: List[str] = []

        def check(field: str, actual: object, expected: object) -> None:
            if actual != expected:
                mismatches.append(
                    f"{field}=artifact({actual!r}) expected({expected!r})"
                )

        check("config_class", self.config_class, other.config_class)
        check("config_name_or_path", self.config_name_or_path, other.config_name_or_path)
        check("model_type", self.model_type, other.model_type)
        check("world_size", self.world_size, other.world_size)
        check("num_hidden_layers", self.num_hidden_layers, other.num_hidden_layers)
        check("num_stages", self.num_stages, other.num_stages)

        if mismatches:
            raise ValueError(
                "Offline artifact metadata mismatch: " + "; ".join(mismatches)
            )
