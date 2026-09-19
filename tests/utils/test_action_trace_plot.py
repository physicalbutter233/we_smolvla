from collections import deque

import torch

from lerobot.utils.action_trace_plot import ActionTraceSampler, AsyncActionTraceSampler
from lerobot.utils.constants import ACTION


class _Policy:
    def __init__(self, queued_actions):
        self._queues = {ACTION: deque(queued_actions)}


def test_action_trace_sampler_records_full_chunk_and_executed_values():
    policy = _Policy([torch.tensor([[2.0, 3.0]]), torch.tensor([[3.0, 4.0]])])
    sampler = ActionTraceSampler(
        action_names=["joint_a.pos", "joint_b.pos"],
        fps=30.0,
        start_time=0.0,
    )

    sample = sampler.sample(
        policy=policy,
        postprocessor=lambda value: value,
        action_values=torch.tensor([[1.0, 2.0]]),
        sent_action={"joint_a.pos": 1.25, "joint_b.pos": 2.25},
        queue_len_before=0,
    )

    assert sample["queue_len"] == 2
    assert sample["chunk_step"] == 0
    assert sample["predicted"] == [1.0, 2.0]
    assert sample["executed"] == [1.25, 2.25]
    assert sample["predicted_chunk"] == [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]

    policy._queues[ACTION].popleft()
    second_sample = sampler.sample(
        policy=policy,
        postprocessor=lambda value: value,
        action_values=torch.tensor([[2.0, 3.0]]),
        sent_action={"joint_a.pos": 2.25, "joint_b.pos": 3.25},
        queue_len_before=2,
    )

    assert second_sample["queue_len"] == 1
    assert second_sample["chunk_step"] == 1
    assert second_sample["predicted"] == [2.0, 3.0]
    assert second_sample["predicted_chunk"] is None


def test_action_trace_sampler_handles_policies_without_action_queue():
    sampler = ActionTraceSampler(
        action_names=["joint_a.pos"],
        fps=30.0,
        start_time=0.0,
    )

    sample = sampler.sample(
        policy=object(),
        postprocessor=lambda value: value,
        action_values=torch.tensor([[5.0]]),
        sent_action={"joint_a.pos": 4.5},
        queue_len_before=0,
    )

    assert sample["queue_len"] == 0
    assert sample["predicted_chunk"] == [[5.0]]


def test_async_action_trace_sampler_records_original_chunk_and_execution():
    sampler = AsyncActionTraceSampler(action_names=["joint_a.pos", "joint_b.pos"])

    chunk_sample = sampler.action_chunk(
        timesteps=[10, 11, 12],
        actions=[
            torch.tensor([1.0, 2.0]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([3.0, 4.0]),
        ],
    )

    assert chunk_sample == {
        "type": "chunk",
        "t": 10.0,
        "timesteps": [10, 11, 12],
        "predicted_chunk": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
    }

    execution_sample = sampler.execution(
        timestep=10,
        queue_len=3,
        executed={"joint_b.pos": 20.0, "joint_a.pos": 10.0},
    )

    assert execution_sample == {
        "type": "execution",
        "t": 10.0,
        "queue_len": 3,
        "executed": [10.0, 20.0],
    }


def test_async_action_trace_sampler_skips_empty_chunks():
    sampler = AsyncActionTraceSampler(action_names=["joint_a.pos"])

    assert sampler.action_chunk(timesteps=[], actions=[]) is None
