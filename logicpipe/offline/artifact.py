from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

from logicpipe.types import OfflineArtifactMetadata, PartitionPlan, ResourceProfile


def save_offline_artifact(
    artifact_path: str,
    profile: ResourceProfile,
    plan: PartitionPlan,
    metadata: OfflineArtifactMetadata,
) -> None:
    payload = {
        "resource_profile": profile.to_dict(),
        "partition_plan": plan.to_dict(),
        "metadata": metadata.to_dict(),
    }
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_offline_artifact(
    artifact_path: str
) -> Tuple[ResourceProfile, PartitionPlan, OfflineArtifactMetadata]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    profile = ResourceProfile.from_dict(payload["resource_profile"])
    plan = PartitionPlan.from_dict(payload["partition_plan"])
    metadata_payload = payload.get("metadata")
    if metadata_payload is None:
        raise ValueError(
            "Offline artifact is missing metadata; delete it to re-profile."
        )
    metadata = OfflineArtifactMetadata.from_dict(metadata_payload)
    return profile, plan, metadata
