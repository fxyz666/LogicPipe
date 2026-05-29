from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class StructuredNode:
    node_id: int
    text: str
    dependencies: List[int]


class StreamingStructuredOutlineParser:
    def __init__(self) -> None:
        self._emitted_ids: Set[int] = set()
        self._pattern = re.compile(r"^\s*(\d+)\.\s*(.+?)\.\s*\[(.*?)\]\s*$")

    def ingest(self, text: str) -> List[StructuredNode]:
        lines = text.splitlines()
        if text and not text.endswith("\n"):
            lines = lines[:-1]

        nodes: List[StructuredNode] = []
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue
            match = self._pattern.match(trimmed)
            if match is None:
                continue
            node_id = int(match.group(1))
            if node_id in self._emitted_ids:
                continue
            deps = _parse_dependency_list(match.group(3))
            if deps is None:
                continue
            nodes.append(
                StructuredNode(
                    node_id=node_id,
                    text=match.group(2).strip(),
                    dependencies=deps,
                )
            )
            self._emitted_ids.add(node_id)
        nodes.sort(key=lambda node: node.node_id)
        return nodes


def get_skeleton_prompt(question: str) -> str:
    prompt = (
        "You're an organizer responsible for only giving the skeleton (not the full content) for answering the question. "
        "Provide the skeleton in a structured list of nodes so we can reconstruct task dependencies. Each line should follow the exact format "
        "`NodeID. BriefContent. [dep1,dep2,...]`, where `NodeID` starts at 0, `BriefContent` is 3-5 short words, and the dependency list "
        "contains zero-based node IDs. Use an empty list (`[]`) when there are no dependencies. Keep the skeleton to 3-10 nodes.\n\n"
        "Question:\nWhat are the typical types of Chinese dishes?\nSkeleton:\n"
        "0. Dumplings. []\n"
        "1. Noodles. []\n"
        "2. Dim Sum. [0,1]\n"
        "3. Hot Pot. [0,1,2]\n"
        "4. Wonton. [0]\n\n"
        "Question:\nWhat are some practical tips for individuals to reduce their carbon emissions?\nSkeleton:\n"
        "0. Energy conservation. []\n"
        "1. Efficient transportation. []\n"
        "2. Home energy efficiency. [0]\n"
        "3. Reduce water consumption. [0,2]\n"
        "4. Sustainable diet. [0]\n"
        "5. Sustainable travel. [1]\n\n"
        "Now, please provide the skeleton for the following question.\n"
        f"{question}\nSkeleton:\n"
    )
    return prompt


def _parse_dependency_list(raw: str) -> Optional[List[int]]:
    raw = raw.strip()
    if not raw:
        return []
    deps = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit():
            return None
        deps.append(int(token))
    return deps


def parse_structured_outline(skeleton: str) -> Optional[List[StructuredNode]]:
    pattern = re.compile(r"^\s*(\d+)\.\s*(.+?)\.\s*\[(.*?)\]\s*$")
    nodes: List[StructuredNode] = []
    for line in skeleton.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        match = pattern.match(trimmed)
        if match is None:
            return None
        node_id = int(match.group(1))
        text = match.group(2).strip()
        deps = _parse_dependency_list(match.group(3))
        if deps is None:
            return None
        nodes.append(StructuredNode(node_id=node_id, text=text, dependencies=deps))
    if not nodes:
        return None
    nodes.sort(key=lambda node: node.node_id)
    return nodes


def get_point_expanding_prompt(
    skeleton: str, question: str
) -> Tuple[List[str], str, List[str], Optional[List[StructuredNode]]]:
    structured_nodes = parse_structured_outline(skeleton)
    if structured_nodes:
        points = [node.text for node in structured_nodes]
    else:
        points = [line.strip() for line in skeleton.splitlines() if line.strip()]
    shared_perfix = get_point_shared_prefix(question, skeleton=skeleton)
    prompts_for_points: List[str] = []
    for idx, point in enumerate(points):
        prompt = f"{idx + 1}. [/INST]"
        prompts_for_points.append(prompt + point)
    return points, shared_perfix, prompts_for_points, structured_nodes


def build_point_prompt(
    node_id: int,
    point_text: str,
    dependency_ids: List[int],
    completed_outputs: Dict[int, str],
) -> str:
    prompt = f"{node_id + 1}. {point_text}"
    dependency_contexts = []
    for dep_id in dependency_ids:
        dep_output = completed_outputs.get(dep_id, "").strip()
        if dep_output:
            dependency_contexts.append(
                f"Dependency node {dep_id}: {dep_output}"
            )
    if dependency_contexts:
        prompt += (
            "\nUse the following completed dependency context while expanding this point:\n"
            + "\n".join(dependency_contexts)
        )
    return f"{prompt} [/INST]"


def build_local_point_prompt(node_id: int, point_text: str) -> str:
    return f"{node_id + 1}. [/INST]{point_text}"


def get_point_shared_prefix(question: str, skeleton: Optional[str] = None) -> str:
    prompt = (
        "[INST] You're responsible for continuing the writing of one and only one point in the overall answer to the following question.\n\n"
        f"{question}\n\n"
    )
    if skeleton:
        prompt += "The skeleton of the answer is\n\n" + skeleton + "\n\n"
    prompt += (
        "Write it **very shortly** in 1~2 sentence and do not continue with other points! Continue and only continue the writing of point "
    )
    return prompt

