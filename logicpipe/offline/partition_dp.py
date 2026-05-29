from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Sequence, Tuple

from logicpipe.types import PartitionPlan, ResourceProfile


_State = Tuple[int, int, int, int]
_PrevState = Optional[_State]


@dataclass
class _DpResult:
    bottleneck_ms: float
    stage_num_hidden_layers_list: List[int]
    selected_devices: List[int]


class JointPartitionSolver:
    """Stage-2 joint device selection and contiguous partitioning.

    This solver follows the paper's DP shape more closely than the previous
    combination-based implementation:
      - it is memory-aware
      - it allows device reuse across multiple stages
      - it tracks the set of used devices for deterministic tie-breaking
    """

    def solve(
        self,
        profile: ResourceProfile,
        num_layers: int,
        num_stages: int,
    ) -> PartitionPlan:
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if num_layers < num_stages:
            raise ValueError("num_layers must be >= num_stages for contiguous blocks")
        if profile.world_size <= 0:
            raise ValueError("profile.world_size must be positive")

        result = self._solve_joint_partition(profile=profile, num_layers=num_layers, num_stages=num_stages)
        return PartitionPlan(
            stage_num_hidden_layers_list=result.stage_num_hidden_layers_list,
            bottleneck_ms=result.bottleneck_ms,
            selected_devices=result.selected_devices,
        )

    def _solve_joint_partition(
        self,
        profile: ResourceProfile,
        num_layers: int,
        num_stages: int,
    ) -> _DpResult:
        world_size = profile.world_size
        layer_cost_prefix = self._prefix_sum(self._normalized_layer_values(profile.layer_compute_ms, num_layers, default=0.0))
        layer_memory_prefix = self._prefix_sum(
            self._normalized_layer_values(profile.layer_memory_mb, num_layers, default=0.0)
        )
        activation_sizes = self._normalized_layer_values(profile.activation_size_mb, num_layers, default=0.0)
        device_slowdown = self._normalized_device_values(profile.device_slowdown, world_size, default=1.0)
        device_memory_mb = self._normalized_device_values(profile.device_memory_mb, world_size, default=inf)
        inter_device_ms = self._normalized_comm_matrix(profile.inter_device_ms, world_size)

        current: Dict[_State, float] = {}
        backpointers: Dict[_State, _PrevState] = {}

        for device_id in range(world_size):
            for end_layer in range(1, num_layers - num_stages + 2):
                segment_memory = self._segment_memory_mb(
                    start_layer=0,
                    end_layer=end_layer,
                    layer_memory_prefix=layer_memory_prefix,
                    activation_sizes=activation_sizes,
                )
                if segment_memory > device_memory_mb[device_id]:
                    continue
                segment_compute = (
                    layer_cost_prefix[end_layer] - layer_cost_prefix[0]
                ) * device_slowdown[device_id]
                state = (1, end_layer, 1 << device_id, device_id)
                existing = current.get(state, inf)
                if segment_compute < existing:
                    current[state] = segment_compute
                    backpointers[state] = None

        for stage_idx in range(2, num_stages + 1):
            next_states: Dict[_State, float] = {}
            next_backpointers: Dict[_State, _PrevState] = {}
            max_end_layer = num_layers - (num_stages - stage_idx)

            for prev_state, prev_bottleneck in current.items():
                _prev_stage_idx, prev_end_layer, prev_mask, prev_device = prev_state
                min_next_end = prev_end_layer + 1
                for end_layer in range(min_next_end, max_end_layer + 1):
                    for device_id in range(world_size):
                        segment_memory = self._segment_memory_mb(
                            start_layer=prev_end_layer,
                            end_layer=end_layer,
                            layer_memory_prefix=layer_memory_prefix,
                            activation_sizes=activation_sizes,
                        )
                        if segment_memory > device_memory_mb[device_id]:
                            continue

                        segment_compute = (
                            layer_cost_prefix[end_layer] - layer_cost_prefix[prev_end_layer]
                        ) * device_slowdown[device_id]
                        comm_ms = inter_device_ms[prev_device][device_id]
                        candidate_bottleneck = max(prev_bottleneck, comm_ms, segment_compute)
                        candidate_mask = prev_mask | (1 << device_id)
                        state = (stage_idx, end_layer, candidate_mask, device_id)
                        existing = next_states.get(state, inf)
                        if candidate_bottleneck < existing:
                            next_states[state] = candidate_bottleneck
                            next_backpointers[state] = prev_state

            if not next_states:
                raise RuntimeError(
                    f"No feasible partition found for stage {stage_idx}; memory budgets are too small."
                )
            current = next_states
            backpointers.update(next_backpointers)

        best_state = self._select_best_terminal_state(current=current, num_layers=num_layers)
        if best_state is None:
            raise RuntimeError("No feasible partition found for the requested number of stages.")

        stage_num_hidden_layers_list, selected_devices = self._reconstruct_partition(
            terminal_state=best_state,
            backpointers=backpointers,
            num_stages=num_stages,
        )
        return _DpResult(
            bottleneck_ms=current[best_state],
            stage_num_hidden_layers_list=stage_num_hidden_layers_list,
            selected_devices=selected_devices,
        )

    def _select_best_terminal_state(
        self,
        current: Dict[_State, float],
        num_layers: int,
    ) -> Optional[_State]:
        best_state: Optional[_State] = None
        best_key: Optional[Tuple[float, int, int]] = None

        for state, bottleneck in current.items():
            _stage_idx, end_layer, used_mask, last_device = state
            if end_layer != num_layers:
                continue
            candidate_key = (bottleneck, self._bit_count(used_mask), last_device)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_state = state
        return best_state

    def _reconstruct_partition(
        self,
        terminal_state: _State,
        backpointers: Dict[_State, _PrevState],
        num_stages: int,
    ) -> Tuple[List[int], List[int]]:
        states: List[_State] = []
        cursor: Optional[_State] = terminal_state
        while cursor is not None:
            states.append(cursor)
            cursor = backpointers.get(cursor)
        states.reverse()

        if len(states) != num_stages:
            raise RuntimeError("Partition reconstruction failed: unexpected stage count.")

        layer_counts: List[int] = []
        selected_devices: List[int] = []
        prev_end_layer = 0
        for _stage_idx, end_layer, _used_mask, device_id in states:
            layer_counts.append(end_layer - prev_end_layer)
            selected_devices.append(device_id)
            prev_end_layer = end_layer
        return layer_counts, selected_devices

    def _segment_memory_mb(
        self,
        start_layer: int,
        end_layer: int,
        layer_memory_prefix: Sequence[float],
        activation_sizes: Sequence[float],
    ) -> float:
        segment_layer_memory = layer_memory_prefix[end_layer] - layer_memory_prefix[start_layer]
        if end_layer <= start_layer:
            return 0.0
        segment_activation = max(activation_sizes[start_layer:end_layer], default=0.0)
        return segment_layer_memory + segment_activation

    def _normalized_layer_values(
        self,
        values: Sequence[float],
        num_layers: int,
        default: float,
    ) -> List[float]:
        normalized = [float(v) for v in values[:num_layers]]
        if len(normalized) < num_layers:
            normalized.extend([default] * (num_layers - len(normalized)))
        return normalized

    def _normalized_device_values(
        self,
        values: Sequence[float],
        world_size: int,
        default: float,
    ) -> List[float]:
        normalized = [float(v) for v in values[:world_size]]
        if len(normalized) < world_size:
            normalized.extend([default] * (world_size - len(normalized)))
        return normalized

    def _normalized_comm_matrix(
        self,
        matrix: Sequence[Sequence[float]],
        world_size: int,
    ) -> List[List[float]]:
        normalized: List[List[float]] = []
        for src in range(world_size):
            if src < len(matrix):
                row = [float(v) for v in matrix[src][:world_size]]
            else:
                row = []
            if len(row) < world_size:
                row.extend([0.0] * (world_size - len(row)))
            if src < world_size:
                row[src] = 0.0
            normalized.append(row)
        return normalized

    def _prefix_sum(self, values: Sequence[float]) -> List[float]:
        prefix = [0.0]
        for value in values:
            prefix.append(prefix[-1] + float(value))
        return prefix

    def _bit_count(self, value: int) -> int:
        return int(value).bit_count()
