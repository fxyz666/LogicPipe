import argparse

from logicpipe.orchestrator import LogicPipeOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal LogicPipe runner")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world", default=2, type=int)
    parser.add_argument(
        "--config_file",
        type=str,
        default="tasks/medusa_llama/config/vicuna_7b_config.json",
        help="Model config path.",
    )
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument(
        "--question",
        type=str,
        default="What are the most effective ways to deal with stress?",
        help="User question for end-to-end LogicPipe inference.",
    )
    parser.add_argument(
        "--offline_artifact",
        type=str,
        default="artifacts/logicpipe/offline_plan.json",
        help="Path to save/load offline profiling and partition artifact.",
    )
    parser.add_argument(
        "--reuse_offline_artifact",
        action="store_true",
        help="Reuse existing offline artifact if available.",
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=None,
        help="Number of pipeline stages to plan for (defaults to `--world`).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_stages is None:
        args.num_stages = args.world
    orchestrator = LogicPipeOrchestrator(
        args=args,
        question=args.question,
        artifact_path=args.offline_artifact,
        reuse_offline_artifact=args.reuse_offline_artifact,
    )
    result = orchestrator.run()
    if result.partition_plan.selected_devices:
        print("Selected devices:", result.partition_plan.selected_devices, flush=True)
    print("Stage partition:", result.partition_plan.stage_num_hidden_layers_list, flush=True)
    print("Estimated bottleneck(ms):", round(result.partition_plan.bottleneck_ms, 3), flush=True)


if __name__ == "__main__":
    main()
