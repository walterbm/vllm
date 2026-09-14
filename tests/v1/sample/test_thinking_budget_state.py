# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder batch index moves."""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


class _MockReasoningConfig:
    reasoning_start_token_ids = [151667]
    reasoning_end_token_ids = [151668]
    thinking_budget_action = "force_end"


def _make_holder() -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        _MockReasoningConfig(),
        8,
        0,
        torch.device("cpu"),
        False,
    )


def test_swap_budgeted_with_unbudgeted_clears_empty_side():
    """Asymmetric SWAP must not leave the empty index sharing state."""
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=5), None, []),
                (1, SamplingParams(), None, []),
            ],
            moved=(),
        )
    )
    assert list(h._state.keys()) == [0]
    budget_state = h._state[0]

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [1]
    assert h._state[1] is budget_state
    assert h._state[1]["thinking_token_budget"] == 5

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [0]
    assert h._state[0] is budget_state


def test_swap_exchanges_two_budgeted_states():
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=3), None, []),
                (1, SamplingParams(thinking_token_budget=7), None, []),
            ],
            moved=(),
        )
    )
    b0 = h._state[0]["thinking_token_budget"]
    b1 = h._state[1]["thinking_token_budget"]
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert h._state[0]["thinking_token_budget"] == b1
    assert h._state[1]["thinking_token_budget"] == b0


class _MockTruncateReasoningConfig(_MockReasoningConfig):
    thinking_budget_action = "truncate"


def test_truncate_mode_reports_exhaustion_instead_of_forcing():
    h = ThinkingBudgetStateHolder(
        _MockTruncateReasoningConfig(), 8, 0, torch.device("cpu"), False
    )
    start = _MockReasoningConfig.reasoning_start_token_ids
    end_id = _MockReasoningConfig.reasoning_end_token_ids[0]
    output: list[int] = []
    over_budget_prompt = [1, *start, 5, 6, 7]
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=2), None, output),
                (1, SamplingParams(thinking_token_budget=2), over_budget_prompt, []),
            ],
            moved=(),
        )
    )
    # Slot 1 is already over budget from its prompt: nothing may be kept.
    h.update_state([output, []], None)
    assert h.take_exhausted() == {1: 0}

    for tok in [*start, 5]:
        output.append(tok)
        h.update_state([output, []], None)
        assert 0 not in h.take_exhausted()

    output.append(6)  # second reasoning token: budget reached
    h.update_state([output, []], None)
    logits = torch.zeros((2, 200_000))
    assert torch.all(h.apply_to_logits(logits, False, None)[:, end_id] == 0)
    assert h.take_exhausted()[0] == 0
    assert h.take_exhausted() == {}


def test_unidirectional_move_of_unbudgeted_request_clears_stale_state():
    """Condensing an unbudgeted request into a freed slot must not leave the
    freed slot's budget state behind."""
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=5), None, []),
                (1, SamplingParams(), None, []),
            ],
            moved=(),
        )
    )
    # Slot 0 finishes; condense() reuses the freed index for slot 1's request
    # via a unidirectional move (the removal itself is consumed by condense).
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=(),
            moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)],
        )
    )
    assert h._state == {}
