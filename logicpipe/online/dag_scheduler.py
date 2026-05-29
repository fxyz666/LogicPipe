from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Set

from core.decoding_pipeline import DecodingPipeline
from tasks.medusa_llama.kv_cache import initialize_past_key_values
from tasks.medusa_llama.outline_decoding_controller import (
    OutlineDecodingController,
    set_controller,
)
from tools.sot import StructuredNode


@dataclass
class DagTask:
    task_id: int
    point_text: str
    dependencies: List[int]


class LogicPipeController(OutlineDecodingController):
    """Controller with explicit DAG metadata and ready queue."""

    def __init__(self, points: List[str], config: Any, model: Any) -> None:
        super().__init__(points, config, model)
        # Parent is singleton-style; refresh runtime bindings for repeated runs.
        self.config = config
        self.model = model
        self.dependencies: Dict[int, Set[int]] = {}
        self.children: Dict[int, Set[int]] = {}
        self.finished: Set[int] = set()
        self.ready_queue: Deque[int] = deque()

    def force_reset_points(self, points: List[str]) -> None:
        # Reinitialize mutable state for repeated runs in same process.
        super().force_reset_points(points)
        self.is_finish = [False] * self.point_num
        self.prepare_point_kv_cache()

    def prepare_point_kv_cache(self) -> None:
        self.past_key_values_for_point = []
        self.past_key_values_data_for_point = []
        self.current_length_data_for_point = []
        for _ in range(self.point_num):
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
                self.model
            )
            self.past_key_values_for_point.append(past_key_values)
            self.past_key_values_data_for_point.append(past_key_values_data)
            self.current_length_data_for_point.append(current_length_data)

    def get_point_past_key_values_data(self, point_id: int):
        return self.past_key_values_data_for_point[point_id]

    def get_outputs(self) -> List[str]:
        tokenizer = self.model.get_tokenizer()
        outputs: List[str] = []
        for i in range(self.point_num):
            input_len = self.input_len_for_point[i]
            input_ids = self.input_ids_for_point[i]
            text = tokenizer.decode(
                input_ids[0, input_len:],
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            outputs.append(text)
        return outputs

    def print_outputs(self) -> None:
        outputs = self.get_outputs()
        for i in range(self.point_num):
            print("********************\n", flush=True)
            print(self.points[i] + outputs[i], flush=True)

    def build_dependency_graph(self, dag_tasks: Iterable[DagTask]) -> None:
        self.dependencies = {idx: set() for idx in range(self.point_num)}
        self.children = {idx: set() for idx in range(self.point_num)}
        for task in dag_tasks:
            self.dependencies[task.task_id] = set(task.dependencies)
            for dep in task.dependencies:
                self.children.setdefault(dep, set()).add(task.task_id)
        self.finished = set()
        self.ready_queue = deque(
            [task_id for task_id, deps in self.dependencies.items() if not deps]
        )

    def pop_ready_point(self) -> Optional[int]:
        if not self.ready_queue:
            return None
        return self.ready_queue.popleft()

    def mark_point_finished(self, point_id: int) -> None:
        if point_id in self.finished:
            return
        self.finished.add(point_id)
        self.is_finish[point_id] = True
        for child in self.children.get(point_id, set()):
            self.dependencies[child].discard(point_id)
            if not self.dependencies[child] and child not in self.finished:
                self.ready_queue.append(child)

    def all_points_finished(self) -> bool:
        return super().all_points_finished()


class LogicAwareDagScheduler:
    """Stage-5 logic-aware scheduling + pipeline decoding wrapper."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._task_map: Dict[int, DagTask] = {}
        self._dispatched_task_ids: Set[int] = set()
        self._completed_task_ids: Set[int] = set()
        self._decoding_pipeline: Optional[DecodingPipeline] = None

    def build_tasks(
        self, points: List[str], structured_outline: Optional[List[StructuredNode]]
    ) -> List[DagTask]:
        if structured_outline:
            tasks: List[DagTask] = []
            seen: Set[int] = set()
            for node in structured_outline:
                if node.node_id in seen:
                    continue
                seen.add(node.node_id)
                text = node.text if node.text else (
                    points[node.node_id] if node.node_id < len(points) else ""
                )
                deps = [dep for dep in node.dependencies if 0 <= dep < len(points)]
                tasks.append(DagTask(task_id=node.node_id, point_text=text, dependencies=deps))
            return tasks
        # Minimal default: chain dependencies to preserve logical order.
        tasks = []
        for idx, point in enumerate(points):
            deps = [idx - 1] if idx > 0 else []
            tasks.append(DagTask(task_id=idx, point_text=point, dependencies=deps))
        return tasks

    def reset_streaming_tasks(self) -> None:
        self._task_map = {}
        self._dispatched_task_ids = set()
        self._completed_task_ids = set()
        self._decoding_pipeline = None

    def register_structured_nodes(
        self,
        nodes: List[StructuredNode],
    ) -> List[DagTask]:
        new_tasks: List[DagTask] = []
        for node in nodes:
            if node.node_id in self._task_map:
                continue
            task = DagTask(
                task_id=node.node_id,
                point_text=node.text,
                dependencies=list(node.dependencies),
            )
            self._task_map[node.node_id] = task
            new_tasks.append(task)
        new_tasks.sort(key=lambda task: task.task_id)
        return new_tasks

    def register_tasks(self, tasks: List[DagTask]) -> List[DagTask]:
        new_tasks: List[DagTask] = []
        for task in tasks:
            if task.task_id in self._task_map:
                continue
            self._task_map[task.task_id] = task
            new_tasks.append(task)
        new_tasks.sort(key=lambda task: task.task_id)
        return new_tasks

    def get_registered_tasks(self) -> List[DagTask]:
        return [self._task_map[task_id] for task_id in sorted(self._task_map)]

    def get_ready_tasks(
        self,
        completed_task_ids: Set[int],
        activated_task_ids: Set[int],
    ) -> List[DagTask]:
        ready_tasks: List[DagTask] = []
        for task in self.get_registered_tasks():
            if task.task_id in activated_task_ids:
                continue
            if all(dep in completed_task_ids for dep in task.dependencies):
                ready_tasks.append(task)
        return ready_tasks

    def mark_task_dispatched(self, task_id: int) -> None:
        self._dispatched_task_ids.add(task_id)

    def mark_task_completed(self, task_id: int) -> None:
        self._completed_task_ids.add(task_id)

    def get_dispatchable_tasks(self) -> List[DagTask]:
        ready_tasks: List[DagTask] = []
        for task_id in sorted(self._task_map):
            if task_id in self._dispatched_task_ids:
                continue
            task = self._task_map[task_id]
            if all(dep in self._completed_task_ids for dep in task.dependencies):
                ready_tasks.append(task)
        return ready_tasks

    def get_completed_task_ids(self) -> Set[int]:
        return set(self._completed_task_ids)

    def build_execution_batches(self, tasks: List[DagTask]) -> List[List[DagTask]]:
        if not tasks:
            return []

        task_map = {task.task_id: task for task in tasks}
        dependencies: Dict[int, Set[int]] = {
            task.task_id: set(task.dependencies) for task in tasks
        }
        children: Dict[int, Set[int]] = {task.task_id: set() for task in tasks}
        for task in tasks:
            for dep in task.dependencies:
                if dep in children:
                    children[dep].add(task.task_id)

        ready = sorted(task_id for task_id, deps in dependencies.items() if not deps)
        batches: List[List[DagTask]] = []
        finished: Set[int] = set()

        while ready:
            current_batch_ids = ready
            batches.append([task_map[task_id] for task_id in current_batch_ids])
            next_ready: List[int] = []
            for task_id in current_batch_ids:
                finished.add(task_id)
                for child in sorted(children.get(task_id, set())):
                    dependencies[child].discard(task_id)
                    if not dependencies[child] and child not in finished:
                        next_ready.append(child)
            ready = sorted(set(next_ready))

        if len(finished) != len(tasks):
            # Fallback to sequential execution if the parsed graph is malformed.
            return [[task] for task in tasks]
        return batches

    def setup_controller(
        self,
        points: List[str],
        point_input_ids: List[Any],
        medusa_logits_list: Optional[List[Any]],
        logits_list: Optional[List[Any]],
        structured_outline: Optional[List[StructuredNode]] = None,
    ) -> LogicPipeController:
        controller = LogicPipeController(points, self.runtime.config, self.runtime.model)
        controller.force_reset_points(points)
        controller.set_up_input_ids_for_point(point_input_ids)
        controller.build_dependency_graph(
            self.build_tasks(points, structured_outline)
        )
        if self.runtime.config.is_last_stage and medusa_logits_list is not None and logits_list is not None:
            # Keep compatibility with current decode pipeline by preloading all requests.
            controller.add_requests(medusa_logits_list, logits_list)
        set_controller(controller)
        return controller

    def run_decoding(self, max_steps: Optional[int] = None) -> int:
        if self._decoding_pipeline is None:
            self.runtime.model.set_mask_for_medusa_decoding()
            self._decoding_pipeline = DecodingPipeline(
                self.runtime.model,
                self.runtime.config,
                self.runtime.args,
            )
        return self._decoding_pipeline.jupiter_decoding_pipeline(max_steps=max_steps)
