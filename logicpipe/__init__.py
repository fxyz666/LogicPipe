"""Minimal LogicPipe refactor skeleton.

This package provides a thin orchestration layer that aligns the current
core implementation with the five-stage LogicPipe workflow.
"""

from .orchestrator import LogicPipeOrchestrator, LogicPipeResult

__all__ = ["LogicPipeOrchestrator", "LogicPipeResult"]
