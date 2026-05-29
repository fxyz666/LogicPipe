from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.utils import NormalDecodingSession, normal_decoding


@dataclass
class BranchDecodeRequest:
    branch_id: int
    prompt: str
    medusa_logits: Any
    logits: Any
    input_ids: Any = None
    static_cache_key: Optional[str] = None
    dependency_cache_keys: List[str] = field(default_factory=list)


@dataclass
class BranchVerificationRound:
    round_id: int
    branch_texts: Dict[int, str]
    accepted_prefix: str
    accepted_delta: str
    active_branch_ids: List[int]
    finished_branch_ids: List[int]


@dataclass
class MultiBranchDecodeResult:
    branch_texts: Dict[int, str]
    verification_rounds: List[BranchVerificationRound]
    accepted_prefix: str


def _longest_common_prefix_text(texts: Iterable[str]) -> str:
    normalized = [text for text in texts if text]
    if not normalized:
        return ""
    prefix = normalized[0]
    for text in normalized[1:]:
        limit = min(len(prefix), len(text))
        match_len = 0
        while match_len < limit and prefix[match_len] == text[match_len]:
            match_len += 1
        prefix = prefix[:match_len]
        if not prefix:
            break
    return prefix


class MultiBranchSelfDraftDecoder:
    """
    Stage-4/5 multi-branch self-draft decoder.

    Current status:
    - introduces explicit branch draft / verify rounds
    - tracks longest-common-prefix (LCP) acceptance across branches
    - keeps the old single-branch path working through the same abstraction

    It is still not the paper-complete MBSD implementation because draft generation
    and batched verification still reuse the repo's current NormalDecodingSession backend.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def create_branch_session(
        self,
        request: BranchDecodeRequest,
    ) -> NormalDecodingSession:
        return NormalDecodingSession(
            prompt=request.prompt,
            model=self.runtime.model,
            config=self.runtime.config,
            medusa_logits=request.medusa_logits,
            logits=request.logits,
            input_ids=request.input_ids,
        )

    def create_skeleton_session(
        self,
        prompt: str,
        medusa_logits: Any,
        logits: Any,
    ) -> NormalDecodingSession:
        return self.create_branch_session(
            BranchDecodeRequest(
                branch_id=0,
                prompt=prompt,
                medusa_logits=medusa_logits,
                logits=logits,
            )
        )

    def decode_branches(
        self,
        requests: List[BranchDecodeRequest],
        on_round_update: Optional[Callable[[BranchVerificationRound], None]] = None,
        max_rounds: Optional[int] = None,
    ) -> MultiBranchDecodeResult:
        if not requests:
            return MultiBranchDecodeResult(
                branch_texts={},
                verification_rounds=[],
                accepted_prefix="",
            )

        sessions = {
            request.branch_id: self.create_branch_session(request)
            for request in requests
        }
        verification_rounds: List[BranchVerificationRound] = []
        accepted_prefix = ""
        round_id = 0

        while True:
            unfinished = [branch_id for branch_id, session in sessions.items() if not session.finished]
            if not unfinished:
                break
            if max_rounds is not None and round_id >= max_rounds:
                break

            for branch_id in unfinished:
                sessions[branch_id].step()

            branch_texts = {
                branch_id: session.text
                for branch_id, session in sessions.items()
            }
            active_branch_ids = [branch_id for branch_id, session in sessions.items() if not session.finished]
            finished_branch_ids = [branch_id for branch_id, session in sessions.items() if session.finished]
            lcp_text = _longest_common_prefix_text(branch_texts.values())
            accepted_delta = lcp_text[len(accepted_prefix):] if lcp_text.startswith(accepted_prefix) else lcp_text
            accepted_prefix = lcp_text
            verification_round = BranchVerificationRound(
                round_id=round_id,
                branch_texts=branch_texts,
                accepted_prefix=accepted_prefix,
                accepted_delta=accepted_delta,
                active_branch_ids=active_branch_ids,
                finished_branch_ids=finished_branch_ids,
            )
            verification_rounds.append(verification_round)
            if on_round_update is not None:
                on_round_update(verification_round)
            round_id += 1

        return MultiBranchDecodeResult(
            branch_texts={branch_id: session.text for branch_id, session in sessions.items()},
            verification_rounds=verification_rounds,
            accepted_prefix=accepted_prefix,
        )

    def decode_skeleton(
        self,
        prompt: str,
        medusa_logits: Any,
        logits: Any,
        on_text_update: Optional[Callable[[str], None]] = None,
    ) -> str:
        request = BranchDecodeRequest(
            branch_id=0,
            prompt=prompt,
            medusa_logits=medusa_logits,
            logits=logits,
        )

        if on_text_update is None:
            text = normal_decoding(
                prompt=prompt,
                model=self.runtime.model,
                config=self.runtime.config,
                medusa_logits=medusa_logits,
                logits=logits,
            )
            return "\n".join([line.lstrip() for line in text.splitlines()])

        def forward_round(round_state: BranchVerificationRound) -> None:
            on_text_update(round_state.branch_texts.get(0, ""))

        result = self.decode_branches(
            requests=[request],
            on_round_update=forward_round,
        )
        text = result.branch_texts.get(0, "")
        return "\n".join([line.lstrip() for line in text.splitlines()])
