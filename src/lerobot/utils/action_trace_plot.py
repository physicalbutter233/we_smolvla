# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Live action-chunk monitoring in a process separate from Rerun."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import queue
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)

DEFAULT_MAX_POINTS = 6000
DEFAULT_MAX_CHUNKS = 500
DEFAULT_WINDOW_S = 60.0
DEFAULT_REFRESH_HZ = 10.0
MPL_CONFIG_DIR = "/tmp/lerobot-matplotlib"


def _to_action_vector(value: Any) -> np.ndarray:
    """Convert a policy action representation to a one-dimensional NumPy array."""
    if isinstance(value, dict):
        return np.asarray([float(v) for v in value.values()], dtype=np.float64)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    return np.asarray(value).reshape(-1).astype(np.float64, copy=False)


def _mapping_to_action_vector(value: dict[str, Any], action_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [float(value.get(name, np.nan)) for name in action_names],
        dtype=np.float64,
    )


def _queue_for_policy(policy: Any) -> deque | None:
    queues = getattr(policy, "_queues", None)
    if not isinstance(queues, dict):
        return None
    action_queue = queues.get(ACTION)
    return action_queue if isinstance(action_queue, deque) else None


@dataclass
class ActionTraceSampler:
    """Build monitor samples from the policy action queue and executed robot action."""

    action_names: list[str]
    fps: float
    start_time: float
    _chunk_step: int = 0
    _predicted_chunk: np.ndarray | None = None

    def queue_length(self, policy: Any) -> int:
        action_queue = _queue_for_policy(policy)
        return len(action_queue) if action_queue is not None else 0

    def sample(
        self,
        *,
        policy: Any,
        postprocessor: Any,
        action_values: Any,
        sent_action: dict[str, Any],
        queue_len_before: int,
    ) -> dict[str, Any]:
        """Create one sample for the plotting process."""
        action_queue = _queue_for_policy(policy)
        queue_len = len(action_queue) if action_queue is not None else 0
        executed = np.asarray(
            [float(sent_action.get(name, np.nan)) for name in self.action_names],
            dtype=np.float64,
        )

        predicted_chunk = None
        if queue_len_before == 0:
            self._chunk_step = 0

            predicted_parts = [_to_action_vector(action_values)]
            if action_queue is not None:
                for queued_action in list(action_queue):
                    predicted_parts.append(_to_action_vector(postprocessor(queued_action)))

            self._predicted_chunk = np.stack(predicted_parts, axis=0)
            predicted_chunk = self._predicted_chunk

        if self._predicted_chunk is None or len(self._predicted_chunk) == 0:
            self._predicted_chunk = executed.reshape(1, -1)

        step = min(self._chunk_step, len(self._predicted_chunk) - 1)
        predicted = self._predicted_chunk[step]
        self._chunk_step += 1

        sample = {
            "type": "sample",
            "t": time.monotonic() - self.start_time,
            "queue_len": queue_len,
            "chunk_step": step,
            "predicted": predicted.tolist(),
            "executed": executed.tolist(),
            "predicted_chunk": predicted_chunk.tolist() if predicted_chunk is not None else None,
        }
        return sample


@dataclass
class AsyncActionTraceSampler:
    """Build monitor samples from events emitted by the asynchronous RobotClient."""

    action_names: list[str]

    def action_chunk(
        self,
        *,
        timesteps: Sequence[int],
        actions: Sequence[Any],
    ) -> dict[str, Any] | None:
        if not timesteps or not actions:
            return None
        if len(timesteps) != len(actions):
            raise ValueError(
                f"Expected the same number of timesteps and actions, got {len(timesteps)} and {len(actions)}."
            )

        predicted_chunk = np.stack([_to_action_vector(action) for action in actions], axis=0)
        return {
            "type": "chunk",
            "t": float(timesteps[0]),
            "timesteps": [int(timestep) for timestep in timesteps],
            "predicted_chunk": predicted_chunk.tolist(),
        }

    def execution(
        self,
        *,
        timestep: int,
        queue_len: int,
        executed: dict[str, Any] | Any,
    ) -> dict[str, Any]:
        if isinstance(executed, dict):
            executed_vector = _mapping_to_action_vector(executed, self.action_names)
        else:
            executed_vector = _to_action_vector(executed)

        return {
            "type": "execution",
            "t": float(timestep),
            "queue_len": int(queue_len),
            "executed": executed_vector.tolist(),
        }


def _select_backend(matplotlib: Any) -> str:
    requested = os.getenv("LEROBOT_ACTION_TRACE_BACKEND")
    if requested:
        matplotlib.use(requested)
        return requested

    if os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"):
        try:
            import tkinter

            root = tkinter.Tk()
            root.withdraw()
            root.destroy()
            matplotlib.use("TkAgg")
            return "TkAgg"
        except Exception:
            pass

    matplotlib.use("Agg")
    return "Agg"


def _plot_worker(
    sample_queue: Any,
    stop_event: Any,
    action_names: list[str],
    fps: float,
    max_points: int,
    max_chunks: int,
    window_s: float,
    refresh_hz: float,
    x_unit: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)
    os.makedirs(MPL_CONFIG_DIR, exist_ok=True)

    import matplotlib

    backend = _select_backend(matplotlib)
    if backend == "Agg":
        logger.warning(
            "No graphical display was detected. The action trace window will run with the Agg backend "
            "and will not be visible."
        )

    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    x_label = "control step" if x_unit == "control_step" else "elapsed time (s)"
    chunk_step = 1.0 if x_unit == "control_step" else 1.0 / fps
    action_rows = max(1, (len(action_names) + 1) // 2)

    fig = plt.figure(figsize=(14, 3.4 + 2.1 * action_rows), num="LeRobot Action Trace")
    fig.suptitle("Action chunks: predicted (dashed) vs executed (solid)")
    grid = fig.add_gridspec(
        1 + action_rows,
        2,
        height_ratios=[1.15, *[1.0] * action_rows],
        hspace=0.45,
        wspace=0.28,
    )

    queue_ax = fig.add_subplot(grid[0, :])
    queue_ax.set_title("Action queue size")
    queue_ax.set_xlabel(x_label)
    queue_ax.set_ylabel("Queue size")
    queue_ax.grid(True, alpha=0.25)
    (queue_line,) = queue_ax.plot([], [], color="tab:blue", linewidth=1.5, label="queue size")
    queue_ax.legend([queue_line], [queue_line.get_label()], loc="upper left")

    palette = plt.get_cmap("tab10")
    actual_lines = []
    predicted_collections = []
    action_axes = []

    for index, name in enumerate(action_names):
        row = 1 + index // 2
        col = index % 2
        ax = fig.add_subplot(grid[row, col])
        ax.set_title(name)
        ax.set_xlabel(x_label)
        ax.set_ylabel("action")
        ax.grid(True, alpha=0.25)

        color = palette(index % 10)
        (actual_line,) = ax.plot(
            [],
            [],
            color=color,
            linewidth=1.8,
            linestyle="-",
            label="executed",
        )
        predicted_collection = LineCollection(
            [],
            colors=[color],
            linewidths=1.1,
            linestyles="dashed",
            alpha=0.55,
            label="predicted",
        )
        ax.add_collection(predicted_collection)
        ax.legend(loc="upper left")

        actual_lines.append(actual_line)
        predicted_collections.append(predicted_collection)
        action_axes.append(ax)

    queue_times: deque[float] = deque(maxlen=max_points)
    queue_lengths: deque[float] = deque(maxlen=max_points)
    actual_times = [deque(maxlen=max_points) for _ in action_names]
    actual_values = [deque(maxlen=max_points) for _ in action_names]
    predicted_segments = [deque(maxlen=max_chunks) for _ in action_names]

    last_draw = 0.0
    latest_t = 0.0
    latest_chunk_len = 1
    plt.ion()
    if backend != "Agg":
        fig.show()

    try:
        while not stop_event.is_set():
            drain_deadline = time.monotonic() + 0.1
            while time.monotonic() < drain_deadline:
                try:
                    sample = sample_queue.get(timeout=0.01)
                except queue.Empty:
                    break
                sample_type = sample.get("type")
                if sample_type == "stop":
                    return

                t = float(sample["t"])
                latest_t = max(latest_t, t)

                if sample_type == "execution":
                    queue_times.append(t)
                    queue_lengths.append(int(sample["queue_len"]))
                    executed = np.asarray(sample["executed"], dtype=np.float64)
                    for index in range(min(len(action_names), len(executed))):
                        actual_times[index].append(t)
                        actual_values[index].append(float(executed[index]))
                    continue

                if sample_type == "chunk":
                    predicted_chunk = sample.get("predicted_chunk")
                    if predicted_chunk is None:
                        continue

                    predicted_chunk_array = np.asarray(predicted_chunk, dtype=np.float64)
                    latest_chunk_len = max(1, len(predicted_chunk_array))
                    timesteps = sample.get("timesteps")
                    if timesteps is None:
                        chunk_times = t + np.arange(len(predicted_chunk_array), dtype=np.float64) * chunk_step
                    else:
                        chunk_times = np.asarray(timesteps, dtype=np.float64)
                        if len(chunk_times) != len(predicted_chunk_array):
                            chunk_times = (
                                t + np.arange(len(predicted_chunk_array), dtype=np.float64) * chunk_step
                            )

                    for index in range(min(len(action_names), predicted_chunk_array.shape[1])):
                        points = np.column_stack((chunk_times, predicted_chunk_array[:, index]))
                        predicted_segments[index].append(points)
                    continue

                if sample_type != "sample":
                    continue

                queue_times.append(t)
                queue_lengths.append(int(sample["queue_len"]))

                executed = np.asarray(sample["executed"], dtype=np.float64)
                predicted_chunk = sample.get("predicted_chunk")

                if predicted_chunk is not None:
                    predicted_chunk_array = np.asarray(predicted_chunk, dtype=np.float64)
                    latest_chunk_len = max(1, len(predicted_chunk_array))
                    chunk_times = t + np.arange(len(predicted_chunk_array), dtype=np.float64) * chunk_step
                    for index in range(min(len(action_names), predicted_chunk_array.shape[1])):
                        points = np.column_stack((chunk_times, predicted_chunk_array[:, index]))
                        predicted_segments[index].append(points)

                for index in range(min(len(action_names), len(executed))):
                    actual_times[index].append(t)
                    actual_values[index].append(float(executed[index]))

            now = time.monotonic()
            if now - last_draw < 1.0 / refresh_hz:
                if not plt.fignum_exists(fig.number):
                    return
                time.sleep(0.01)
                continue

            last_draw = now
            queue_line.set_data(queue_times, queue_lengths)

            window_left = max(0.0, latest_t - window_s)
            window_right = max(
                latest_t + latest_chunk_len * chunk_step + 0.25,
                window_left + 1.0,
            )
            queue_ax.set_xlim(window_left, window_right)
            queue_ax.set_ylim(0, max(1.0, max(queue_lengths, default=0) * 1.15))

            for index, ax in enumerate(action_axes):
                actual_lines[index].set_data(actual_times[index], actual_values[index])
                predicted_collections[index].set_segments(list(predicted_segments[index]))
                ax.set_xlim(window_left, window_right)

                values = list(actual_values[index])
                for segment in predicted_segments[index]:
                    values.extend(segment[:, 1].tolist())
                if values:
                    low = min(values)
                    high = max(values)
                    pad = max(1e-6, (high - low) * 0.12)
                    ax.set_ylim(low - pad, high + pad)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    finally:
        plt.close(fig)


class ActionTracePlot:
    """Write policy action traces to a Matplotlib process independent of Rerun."""

    def __init__(
        self,
        *,
        action_names: list[str],
        fps: float,
        max_points: int = DEFAULT_MAX_POINTS,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        window_s: float = DEFAULT_WINDOW_S,
        refresh_hz: float = DEFAULT_REFRESH_HZ,
        x_unit: str = "seconds",
    ) -> None:
        if x_unit not in {"seconds", "control_step"}:
            raise ValueError(f"Unknown action trace x_unit: {x_unit}")

        self.action_names = list(action_names)
        self.fps = float(fps)
        self.x_unit = x_unit
        self._sampler = ActionTraceSampler(
            action_names=self.action_names,
            fps=self.fps,
            start_time=time.monotonic(),
        )
        self._async_sampler = AsyncActionTraceSampler(action_names=self.action_names)
        self._closed = False
        self._context = mp.get_context("spawn")
        self._queue = self._context.Queue(maxsize=256)
        self._stop_event = self._context.Event()
        self._process = self._context.Process(
            target=_plot_worker,
            args=(
                self._queue,
                self._stop_event,
                self.action_names,
                self.fps,
                max_points,
                max_chunks,
                window_s,
                refresh_hz,
                self.x_unit,
            ),
            name="lerobot-action-trace",
            daemon=True,
        )
        self._process.start()

    def _put_latest(self, sample: dict[str, Any] | None) -> None:
        if self._closed or sample is None:
            return
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(sample)

    def queue_length(self, policy: Any) -> int:
        return self._sampler.queue_length(policy)

    def record_action(
        self,
        *,
        policy: Any,
        postprocessor: Any,
        action_values: Any,
        sent_action: dict[str, Any],
        queue_len_before: int,
    ) -> None:
        if self._closed:
            return

        sample = self._sampler.sample(
            policy=policy,
            postprocessor=postprocessor,
            action_values=action_values,
            sent_action=sent_action,
            queue_len_before=queue_len_before,
        )
        self._put_latest(sample)

    def record_action_chunk(
        self,
        *,
        timesteps: Sequence[int],
        actions: Sequence[Any],
    ) -> None:
        """Record the original model-predicted chunk before queue aggregation."""
        self._put_latest(
            self._async_sampler.action_chunk(
                timesteps=timesteps,
                actions=actions,
            )
        )

    def record_execution(
        self,
        *,
        timestep: int,
        queue_len: int,
        executed: dict[str, Any] | Any,
    ) -> None:
        """Record the actual action sent to the robot and the queue size before dequeue."""
        self._put_latest(
            self._async_sampler.execution(
                timestep=timestep,
                queue_len=queue_len,
                executed=executed,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait({"type": "stop"})
        self._stop_event.set()
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
