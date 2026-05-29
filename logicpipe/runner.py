from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from logicpipe.context import RuntimeLaunchArgs
from logicpipe.types import PartitionPlan
from tasks.medusa_llama.llama_config import LlamaConfig
from tools.utils import get_model_type, initialize_distributed

SUPPORTED_MODEL_TYPES = {"vicuna_7b", "vicuna_13b"}


def _validate_launch_arguments(args: RuntimeLaunchArgs) -> None:
    if not isinstance(args.world, int):
        raise TypeError("`world` must be an integer specifying the total number of ranks.")
    if args.world <= 0:
        raise ValueError("`world` must be greater than 0.")
    if not isinstance(args.rank, int):
        raise TypeError("`rank` must be an integer.")
    if args.rank < 0 or args.rank >= args.world:
        raise ValueError(f"`rank` ({args.rank}) must satisfy 0 <= rank < world ({args.world}).")
    if not isinstance(args.config_file, str) or not args.config_file:
        raise ValueError("`config_file` must point to a non-empty model config path.")
    num_stages = getattr(args, "num_stages", args.world)
    if not isinstance(num_stages, int):
        raise TypeError("`num_stages` must be an integer when provided.")
    if num_stages <= 0:
        raise ValueError("`num_stages` must be greater than 0.")
    if num_stages != args.world:
        raise ValueError(
            "The current runtime requires `num_stages == world`. "
            "Planning fewer stages than ranks is not wired through runtime rank mapping yet."
        )


def _ensure_supported_model_type(args: RuntimeLaunchArgs) -> str:
    model_type = get_model_type(args.config_file)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type `{model_type}` for {args.config_file}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_MODEL_TYPES))}."
        )
    return model_type


def _validate_partition_plan(partition_plan: Optional[PartitionPlan], world: int) -> None:
    if partition_plan is None:
        return
    stage_count = len(partition_plan.stage_num_hidden_layers_list)
    if stage_count != world:
        raise ValueError(
            f"Partition plan stage count ({stage_count}) must match `world` ({world})."
        )
    if stage_count == 0:
        raise ValueError("Partition plan must declare at least one stage.")


def _resolve_stage_weights_path(model_type: str, world: int, rank: int) -> Path:
    stage_path = Path(f"temp_{model_type}_world_{world}_rank_{rank}") / "stage.bin"
    if not stage_path.is_file():
        raise FileNotFoundError(
            f"Stage weights missing at {stage_path}. Run the model partition tool before starting."
        )
    return stage_path


def _validate_stage_weights_contents(stage_path: Path, config: Any) -> None:
    state_dict = torch.load(stage_path, map_location="cpu")
    layer_prefixes = set()
    for key in state_dict:
        if key.startswith("model.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_prefixes.add(int(parts[2]))
    if len(layer_prefixes) != config.num_pp_hidden_layers:
        raise ValueError(
            f"Stage weights at {stage_path} are incomplete: expected "
            f"{config.num_pp_hidden_layers} local layers, found {len(layer_prefixes)}."
        )
    if config.is_first_stage and "model.embed_tokens.weight" not in state_dict:
        raise ValueError(
            f"First-stage weights at {stage_path} are incomplete: missing embedding weights."
        )
    if config.is_last_stage and "lm_head.weight" not in state_dict:
        raise ValueError(
            f"Last-stage weights at {stage_path} are incomplete: missing LM head weights."
        )


def build_runtime(
    args: RuntimeLaunchArgs,
    config: Any = None,
    partition_plan: Optional[PartitionPlan] = None,
) -> tuple[Any, Any, Any]:
    _validate_launch_arguments(args)
    model_type = _ensure_supported_model_type(args)

    if config is None:
        config = LlamaConfig.from_pretrained(args.config_file)
    _validate_partition_plan(partition_plan, args.world)
    if partition_plan is not None:
        config.stage_num_hidden_layers_list = list(
            partition_plan.stage_num_hidden_layers_list
        )

    initialize_distributed(config, args)
    config.update_pp_stage_config(args)

    stage_path = _resolve_stage_weights_path(model_type, args.world, args.rank)
    _validate_stage_weights_contents(stage_path, config)
    from tasks.medusa_llama.medusa_llama_pp import (
        PPMedusaLlamaForCausalLM as PPMedusaModel,
    )

    if config.device == "cuda":
        with torch.device("cuda"):
            model = PPMedusaModel.from_pretrained(
                pretrained_model_name_or_path=str(stage_path),
                config=config,
                use_safetensors=False,
                torch_dtype=config.torch_dtype,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
            )
    else:
        model = PPMedusaModel.from_pretrained(
            pretrained_model_name_or_path=str(stage_path),
            config=config,
            use_safetensors=False,
            torch_dtype=config.torch_dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )

    model.eval()
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(config.device)
    tokenizer = model.get_tokenizer()
    return config, model, tokenizer
