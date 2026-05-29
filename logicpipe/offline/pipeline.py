from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from logicpipe.offline.artifact import load_offline_artifact, save_offline_artifact
from logicpipe.offline.partition_dp import JointPartitionSolver
from logicpipe.offline.profiler import OfflineResourceProfiler
from logicpipe.types import (
    OfflineArtifactMetadata,
    PartitionPlan,
    ResourceProfile,
)


class OfflinePipelinePlanner:
    """Runs LogicPipe stage-1 and stage-2 in a minimal form."""

    def __init__(self) -> None:
        self.profiler = OfflineResourceProfiler()
        self.solver = JointPartitionSolver()

    def run(
        self,
        config: Any,
        world_size: int,
        num_stages: int,
        artifact_path: Optional[str] = None,
        reuse_artifact: bool = False,
    ) -> Tuple[ResourceProfile, PartitionPlan]:
        num_layers = int(config.num_hidden_layers)
        expected_metadata = OfflineArtifactMetadata.from_config(
            config=config,
            world_size=world_size,
            num_hidden_layers=num_layers,
            num_stages=num_stages,
        )

        artifact_exists = bool(artifact_path) and Path(artifact_path).is_file()
        if reuse_artifact and artifact_exists and artifact_path:
            profile, plan, metadata = load_offline_artifact(artifact_path)
            try:
                metadata.assert_compatible_with(expected_metadata)
            except ValueError as err:
                raise ValueError(
                    f"Offline artifact {artifact_path} is incompatible: {err}"
                ) from err
            return profile, plan
        profile = self.profiler.profile(config=config, world_size=world_size)
        plan = self.solver.solve(
            profile=profile,
            num_layers=num_layers,
            num_stages=num_stages,
        )

        if artifact_path:
            try:
                save_offline_artifact(
                    artifact_path=artifact_path,
                    profile=profile,
                    plan=plan,
                    metadata=expected_metadata,
                )
            except OSError:
                # Artifact persistence is an optimization only; runtime planning can proceed
                # even when the cache path is unavailable in the current environment.
                pass
        return profile, plan
