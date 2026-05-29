from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from core.utils import (
    jupiter_prefilling,
    jupiter_prefilling_no_finish,
    point_prefilling,
)


@dataclass
class PrefillOutput:
    medusa_logits: Any
    logits: Any


class IntraSequencePrefillEngine:
    """Stage-3 intra-sequence parallel partitioning wrapper."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def prefill_prompt(self, prompt: str) -> PrefillOutput:
        input_ids = self.runtime.tokenizer.encode(prompt, return_tensors="pt")
        medusa_logits, logits = jupiter_prefilling(
            input_ids=input_ids,
            model=self.runtime.model,
            config=self.runtime.config,
            args=self.runtime.args,
        )
        dist.barrier()
        return PrefillOutput(medusa_logits=medusa_logits, logits=logits)

    def prefill_shared_prefix(self, shared_prefix: str) -> None:
        input_ids = self.runtime.tokenizer.encode(shared_prefix, return_tensors="pt")
        jupiter_prefilling_no_finish(
            input_ids=input_ids,
            model=self.runtime.model,
            config=self.runtime.config,
            args=self.runtime.args,
        )
        dist.barrier()

    def prefill_points(
        self,
        prompts_for_points: Sequence[str],
        point_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
        medusa_logits_list, logits_list = point_prefilling(
            points=list(prompts_for_points),
            model=self.runtime.model,
            config=self.runtime.config,
            args=self.runtime.args,
            point_ids=list(point_ids) if point_ids is not None else None,
        )
        dist.barrier()
        return medusa_logits_list, logits_list

    def build_point_input_ids(self, shared_prefix: str, point_prompt: str) -> torch.Tensor:
        input_ids_1 = self.runtime.tokenizer.encode(shared_prefix, return_tensors="pt")
        input_ids_2 = self.runtime.tokenizer.encode(point_prompt, return_tensors="pt")
        input_ids = torch.cat([input_ids_1, input_ids_2[:, 2:]], dim=1)
        if self.runtime.config.device == "cuda":
            input_ids = input_ids.cuda()
        return input_ids
