from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from core.decoding_pipeline import DecodingPipeline
from logicpipe.context import LogicPipeRuntime
from logicpipe.offline.pipeline import OfflinePipelinePlanner
from logicpipe.online.branch_decoder import MultiBranchSelfDraftDecoder
from logicpipe.online.dag_scheduler import DagTask, LogicAwareDagScheduler, LogicPipeController
from logicpipe.online.prefill_engine import IntraSequencePrefillEngine
from logicpipe.runner import build_runtime
from logicpipe.types import PartitionPlan
from tasks.medusa_llama.kv_cache import ContextualCacheUnit, load_contextual_cache, snapshot_contextual_cache
from tasks.medusa_llama.llama_config import LlamaConfig
from tasks.medusa_llama.outline_decoding_controller import set_controller
from tools.sot import (
    StreamingStructuredOutlineParser,
    build_local_point_prompt,
    get_point_expanding_prompt,
    get_point_shared_prefix,
    get_skeleton_prompt,
)


@dataclass
class LogicPipeResult:
    question: str
    skeleton: str
    points: List[str]
    outputs: List[str]
    partition_plan: PartitionPlan


class LogicPipeOrchestrator:
    """Runnable LogicPipe orchestrator with streaming DAG registration and KV-level reuse."""

    def __init__(
        self,
        args,
        question: str,
        artifact_path: Optional[str] = None,
        reuse_offline_artifact: bool = False,
    ) -> None:
        self.args = args
        self.question = question
        self.artifact_path = artifact_path
        self.reuse_offline_artifact = reuse_offline_artifact

    def run(self) -> LogicPipeResult:
        config = LlamaConfig.from_pretrained(self.args.config_file)
        offline_planner = OfflinePipelinePlanner()
        num_stages = getattr(self.args, "num_stages", None)
        if num_stages is None:
            num_stages = self.args.world
        if num_stages != self.args.world:
            raise ValueError(
                "The current end-to-end LogicPipe runtime requires `num_stages == world`. "
                "Planning fewer stages than ranks is not supported until runtime rank mapping is implemented."
            )
        _, partition_plan = offline_planner.run(
            config=config,
            world_size=self.args.world,
            num_stages=num_stages,
            artifact_path=self.artifact_path,
            reuse_artifact=self.reuse_offline_artifact,
        )

        config, model, tokenizer = build_runtime(
            self.args,
            config=config,
            partition_plan=partition_plan,
        )
        runtime = LogicPipeRuntime(
            args=self.args,
            config=config,
            model=model,
            tokenizer=tokenizer,
        )

        prefill_engine = IntraSequencePrefillEngine(runtime)
        scheduler = LogicAwareDagScheduler(runtime)
        scheduler.reset_streaming_tasks()
        controller = LogicPipeController([], runtime.config, runtime.model)
        set_controller(controller)
        decode_runtime: Optional[DecodingPipeline] = None
        streaming_parser = StreamingStructuredOutlineParser()

        skeleton_prompt = get_skeleton_prompt(self.question)
        prompt_prefill = prefill_engine.prefill_prompt(skeleton_prompt)
        skeleton_session = MultiBranchSelfDraftDecoder(runtime).create_skeleton_session(
            prompt=skeleton_prompt,
            medusa_logits=prompt_prefill.medusa_logits,
            logits=prompt_prefill.logits,
        )
        planning_cache = snapshot_contextual_cache(runtime.model.past_key_values)

        shared_prefix = get_point_shared_prefix(self.question)
        prefill_engine.prefill_shared_prefix(shared_prefix)
        shared_prefix_cache = snapshot_contextual_cache(runtime.model.past_key_values)

        completed_outputs: Dict[int, str] = {}
        global_contextual_cache: Dict[int, ContextualCacheUnit] = {}
        skeleton = ""
        points: List[str] = []
        planning_finalized = False

        def sync_work_flags() -> List[bool]:
            flags = torch.zeros(2, dtype=torch.int64)
            if runtime.config.is_last_stage:
                flags[0] = 0 if skeleton_session.finished else 1
                flags[1] = 1 if controller.has_pending_requests() else 0
            dist.broadcast(flags, src=runtime.config.total_stage - 1)
            return [bool(flags[0].item()), bool(flags[1].item())]

        def register_tasks(tasks: List[DagTask]) -> None:
            tasks = scheduler.register_tasks(tasks)
            for task in tasks:
                controller.register_point(task.task_id, task.point_text, task.dependencies)

        def register_streaming_nodes_from_text(text: str) -> None:
            new_nodes = streaming_parser.ingest(text)
            if not new_nodes:
                return
            register_tasks(scheduler.register_structured_nodes(new_nodes))

        def activate_ready_tasks() -> int:
            ready_tasks = scheduler.get_dispatchable_tasks()
            if not ready_tasks:
                return 0

            load_contextual_cache(runtime.model.past_key_values, shared_prefix_cache)
            point_ids = [task.task_id for task in ready_tasks]
            point_prompts = [
                build_local_point_prompt(
                    node_id=task.task_id,
                    point_text=task.point_text,
                )
                for task in ready_tasks
            ]

            for task in ready_tasks:
                controller.register_point(task.task_id, task.point_text, task.dependencies)
                dependency_cache_units = [
                    global_contextual_cache[dep_id]
                    for dep_id in task.dependencies
                    if dep_id in global_contextual_cache
                ]
                if dependency_cache_units:
                    controller.inject_contextual_caches(task.task_id, dependency_cache_units)

            medusa_logits_list, logits_list = prefill_engine.prefill_points(
                point_prompts,
                point_ids=point_ids,
            )

            for task, point_prompt in zip(ready_tasks, point_prompts):
                point_input_ids = prefill_engine.build_point_input_ids(shared_prefix, point_prompt)
                controller.set_input_ids_for_point(task.task_id, point_input_ids)
                scheduler.mark_task_dispatched(task.task_id)

            if runtime.config.is_last_stage and medusa_logits_list is not None and logits_list is not None:
                for task, medusa_logits, logits in zip(ready_tasks, medusa_logits_list, logits_list):
                    controller.add_request(medusa_logits, logits, task.task_id)
            return len(ready_tasks)

        def collect_finished_points() -> int:
            finished_point_ids = controller.drain_finished_points()
            for point_id in finished_point_ids:
                scheduler.mark_task_completed(point_id)
                completed_outputs[point_id] = controller.get_output_for_point(point_id)
                global_contextual_cache[point_id] = controller.export_point_contextual_cache(point_id)
            return len(finished_point_ids)

        while True:
            planning_active, decode_active = sync_work_flags()

            if planning_active:
                load_contextual_cache(runtime.model.past_key_values, planning_cache)
                runtime.model.set_mask_for_medusa_decoding()
                skeleton = skeleton_session.step()
                planning_cache = snapshot_contextual_cache(runtime.model.past_key_values)
                register_streaming_nodes_from_text(skeleton)
                activate_ready_tasks()

                if skeleton_session.finished and not planning_finalized:
                    points, _, _, structured_outline = get_point_expanding_prompt(
                        skeleton,
                        self.question,
                    )
                    if structured_outline:
                        register_tasks(scheduler.register_structured_nodes(structured_outline))
                    else:
                        register_tasks(scheduler.build_tasks(points, structured_outline))
                    planning_finalized = True
                    activate_ready_tasks()

            if decode_active:
                if decode_runtime is None:
                    decode_runtime = DecodingPipeline(runtime.model, runtime.config, self.args)
                load_contextual_cache(runtime.model.past_key_values, shared_prefix_cache)
                runtime.model.set_mask_for_medusa_decoding()
                decode_runtime.jupiter_decoding_step()
                collect_finished_points()
                activate_ready_tasks()

            if planning_finalized and controller.all_points_finished():
                break

            if not planning_active and not decode_active and planning_finalized:
                raise RuntimeError(
                    "LogicPipe runtime became idle before all registered points finished. "
                    "The DAG may contain unresolved dependencies or a point was never dispatched."
                )

        tasks = scheduler.get_registered_tasks()
        outputs = [completed_outputs.get(task.task_id, "") for task in tasks]
        if runtime.config.is_last_stage:
            for task in tasks:
                print("********************\n", flush=True)
                print(task.point_text + completed_outputs.get(task.task_id, ""), flush=True)

        return LogicPipeResult(
            question=self.question,
            skeleton=skeleton,
            points=points,
            outputs=outputs,
            partition_plan=partition_plan,
        )
