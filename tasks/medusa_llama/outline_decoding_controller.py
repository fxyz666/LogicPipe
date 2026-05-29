from core.core.threadsafe_queue import Queue
from .kv_cache import (
    ContextualCacheUnit,
    append_contextual_cache_unit,
    initialize_past_key_values,
    snapshot_contextual_cache,
)
from typing import Any, Dict, List, Optional, Set


class OutlineDecodingController:
    """Base controller that exposes a dependency-aware request queue."""

    def __init__(self, points: List[str], config: Any, model: Any) -> None:
        self.points = points
        self.point_num = len(points)
        self.config = config
        self.model = model
        self.past_key_values_for_point: List[Any] = []
        self.past_key_values_data_for_point: List[Any] = []
        self.current_length_data_for_point: List[Any] = []
        self.input_ids_for_point: List[Any] = []
        self.input_len_for_point: List[int] = []
        self.request_queue: Optional[Queue] = Queue() if config.is_last_stage else None
        self.point_requests: List[Optional[Dict[str, Any]]] = [None] * self.point_num
        self.points_in_queue: Set[int] = set()
        self.dependencies: Dict[int, Set[int]] = {idx: set() for idx in range(self.point_num)}
        self.children: Dict[int, Set[int]] = {idx: set() for idx in range(self.point_num)}
        self.finished: Set[int] = set()
        self.finished_order: List[int] = []
        self.is_finish = [False] * self.point_num
        self.prepare_point_kv_cache()

    def force_reset_points(self, points: List[str]) -> None:
        self.points = list(points)
        self.point_num = len(points)
        self.point_requests = [None] * self.point_num
        self.points_in_queue.clear()
        self.dependencies = {idx: set() for idx in range(self.point_num)}
        self.children = {idx: set() for idx in range(self.point_num)}
        self.finished.clear()
        self.finished_order = []
        self.is_finish = [False] * self.point_num
        if self.config.is_last_stage:
            self.request_queue = Queue()
        self.input_ids_for_point = []
        self.input_len_for_point = []
        self.past_key_values_for_point = []
        self.past_key_values_data_for_point = []
        self.current_length_data_for_point = []

    def _append_point_slot(self, point_text: str) -> int:
        point_id = self.point_num
        self.points.append(point_text)
        self.point_num += 1
        self.point_requests.append(None)
        self.is_finish.append(False)
        self.input_ids_for_point.append(None)
        self.input_len_for_point.append(0)
        self.dependencies[point_id] = set()
        self.children[point_id] = set()
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(self.model)
        self.past_key_values_for_point.append(past_key_values)
        self.past_key_values_data_for_point.append(past_key_values_data)
        self.current_length_data_for_point.append(current_length_data)
        return point_id

    def ensure_point_slot(self, point_id: int, point_text: str) -> None:
        while self.point_num <= point_id:
            next_text = point_text if self.point_num == point_id else f"Node {self.point_num}"
            self._append_point_slot(next_text)
        self.points[point_id] = point_text

    def register_point(
        self,
        point_id: int,
        point_text: str,
        dependencies: Optional[List[int]] = None,
    ) -> None:
        self.ensure_point_slot(point_id, point_text)
        if dependencies is None:
            dependencies = []
        self.dependencies[point_id] = set(dep for dep in dependencies if dep != point_id)
        for dep in dependencies:
            self.children.setdefault(dep, set()).add(point_id)

    def set_input_ids_for_point(self, point_id: int, input_ids: Any) -> None:
        self.ensure_point_slot(point_id, self.points[point_id] if point_id < len(self.points) else f"Node {point_id}")
        self.input_ids_for_point[point_id] = input_ids
        self.input_len_for_point[point_id] = input_ids.shape[1]

    def set_up_input_ids_for_point(self, input_ids_for_point: List[Any]) -> None:
        self.input_ids_for_point = input_ids_for_point
        self.input_len_for_point = [input_ids.shape[1] for input_ids in input_ids_for_point]

    def _enqueue_point_if_ready(self, point_id: int) -> None:
        if not self.config.is_last_stage or self.request_queue is None:
            return
        if point_id in self.finished or point_id in self.points_in_queue:
            return
        if self.point_requests[point_id] is None:
            return
        self.request_queue.add(point_id)
        self.points_in_queue.add(point_id)

    def add_request(self, medusa_logits: Any, logits: Any, point_id: int) -> None:
        assert self.config.is_last_stage
        self.point_requests[point_id] = {
            "medusa_logits": medusa_logits,
            "logits": logits,
        }
        self._enqueue_point_if_ready(point_id)

    def add_requests(self, medusa_logits_list: List[Any], logits_list: List[Any]) -> None:
        assert self.config.is_last_stage
        for point_id in range(self.point_num):
            self.point_requests[point_id] = {
                "medusa_logits": medusa_logits_list[point_id],
                "logits": logits_list[point_id],
            }
        for point_id in range(self.point_num):
            if not self.dependencies.get(point_id):
                self._enqueue_point_if_ready(point_id)

    def get_request(self) -> Dict[str, Any]:
        assert self.config.is_last_stage and self.request_queue is not None
        point_id = self.request_queue.remove()
        self.points_in_queue.discard(point_id)
        payload = self.point_requests[point_id]
        if payload is None:
            raise RuntimeError(f"Missing request payload for point {point_id}")
        return {
            "point_id": point_id,
            "medusa_logits": payload["medusa_logits"],
            "logits": payload["logits"],
        }

    def has_pending_requests(self) -> bool:
        if not self.config.is_last_stage or self.request_queue is None:
            return False
        return self.request_queue.len() > 0

    def finalize_point(self, point_id: int) -> None:
        if point_id in self.finished:
            return
        self.finished.add(point_id)
        self.finished_order.append(point_id)
        self.is_finish[point_id] = True
        self.points_in_queue.discard(point_id)
        for child in self.children.get(point_id, set()):
            self.dependencies.setdefault(child, set()).discard(point_id)
            if not self.dependencies[child]:
                self._enqueue_point_if_ready(child)

    def mark_point_finished(self, point_id: int) -> None:
        self.finalize_point(point_id)

    def all_points_finished(self) -> bool:
        return len(self.finished) >= self.point_num

    def drain_finished_points(self) -> List[int]:
        finished = list(self.finished_order)
        self.finished_order = []
        return finished

    def prepare_point_kv_cache(self) -> None:
        print("=====================\n prepare point kv cache")
        for _ in range(self.point_num):
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(self.model)
            self.past_key_values_for_point.append(past_key_values)
            self.past_key_values_data_for_point.append(past_key_values_data)
            self.current_length_data_for_point.append(current_length_data)

    def get_point_past_key_values_data(self, point_id: int):
        return self.past_key_values_data_for_point[point_id]

    def get_point_past_key_values(self, point_id: int):
        return self.past_key_values_for_point[point_id]

    def get_point_current_length_data(self, point_id: int):
        return self.current_length_data_for_point[point_id]

    def get_point_kv_length(self, point_id: int) -> int:
        current_length_data = self.get_point_current_length_data(point_id)
        return int(current_length_data[0].item())

    def export_point_contextual_cache(self, point_id: int) -> ContextualCacheUnit:
        return snapshot_contextual_cache(self.get_point_past_key_values(point_id))

    def inject_contextual_caches(
        self,
        point_id: int,
        cache_units: List[ContextualCacheUnit],
    ) -> None:
        point_past_key_values = self.get_point_past_key_values(point_id)
        for cache_unit in cache_units:
            append_contextual_cache_unit(point_past_key_values, cache_unit)

    def get_input_ids(self, point_id: int):
        return self.input_ids_for_point[point_id]

    def update_input_ids(self, input_ids, point_id: int):
        self.input_ids_for_point[point_id] = input_ids

    def get_input_len(self, point_id: int):
        return self.input_len_for_point[point_id]

    def get_output_for_point(self, point_id: int) -> str:
        tokenizer = self.model.get_tokenizer()
        input_len = self.input_len_for_point[point_id]
        input_ids = self.input_ids_for_point[point_id]
        return tokenizer.decode(
            input_ids[0, input_len:],
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

    def get_finished_points(self) -> List[int]:
        return sorted(self.finished)

    def get_output(self):
        tokenizer = self.model.get_tokenizer()
        for i in range(self.point_num):
            input_len = self.input_len_for_point[i]
            input_ids = self.input_ids_for_point[i]
            text = tokenizer.decode(
                input_ids[0, input_len:],
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            print("********************\n", flush=True)
            print(self.points[i] + text, flush=True)

global controller


def get_controller():
    if controller is None:
        raise RuntimeError("Outline decoding controller has not been initialized.")
    return controller


def set_controller(con):
    global controller
    controller = con
controller = None
