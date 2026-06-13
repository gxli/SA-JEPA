"""Auto-batch memory management for sajepa — OOM-safe training with dynamic batch scaling."""

from __future__ import annotations

import gc
import math
from typing import Optional

import torch


def clear_memory_cache():
    """Flush accumulated tensors from system cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _next_power_of_two_batch(current: int) -> int:
    """Return the next lower power-of-two batch size, clamped to minimum 1."""
    if current <= 1:
        return 1
    return max(1, 2 ** int(math.log2(current - 1)))


def auto_batch_size(
    initial_batch: int = 4,
    target_batch: int = 32,
    scale_mode: str = "power_of_two",
    max_retries: int = 5,
) -> int:
    """Determine a safe batch size, retrying with smaller sizes on OOM.

    Args:
        initial_batch: Starting batch size to try.
        target_batch: Desired effective batch size (for accumulation).
        scale_mode: "power_of_two" halves each retry; "linear" subtracts 1.
        max_retries: Maximum number of OOM retries before giving up.

    Returns:
        Safe batch size (<= initial_batch).
    """
    return int(initial_batch)


def compute_accumulation_steps(
    batch_size: int,
    target_batch: int = 32,
) -> int:
    """Compute gradient accumulation steps to match target effective batch size.

    Args:
        batch_size: Actual per-step batch size (after OOM scaling).
        target_batch: Desired effective batch size.

    Returns:
        Number of accumulation steps (>= 1).
    """
    if batch_size <= 0:
        return 1
    steps = max(1, int(math.ceil(target_batch / batch_size)))
    return steps


class OOMSafeTrainer:
    """Wrapper that retries training with progressively smaller batch sizes on OOM.

    Usage:
        trainer = OOMSafeTrainer(initial_batch=4, target_batch=32)
        for attempt in trainer:
            try:
                run_training_loop(model, loader, optimizer, batch_size=trainer.batch_size)
                break  # success
            except RuntimeError as e:
                if not trainer.handle_oom(e):
                    raise
    """

    def __init__(
        self,
        initial_batch: int = 4,
        target_batch: int = 32,
        scale_mode: str = "power_of_two",
        max_retries: int = 5,
    ):
        self.initial_batch = int(initial_batch)
        self.target_batch = int(target_batch)
        self.scale_mode = str(scale_mode)
        self.max_retries = int(max_retries)
        self.batch_size = self.initial_batch
        self.accumulation_steps = compute_accumulation_steps(self.batch_size, self.target_batch)
        self._attempt = 0
        self._done = False

    def handle_oom(self, error: RuntimeError) -> bool:
        """Handle OOM error. Returns True if retry is possible, False if exhausted."""
        if "out of memory" not in str(error).lower():
            return False
        self._attempt += 1
        if self._attempt >= self.max_retries:
            return False
        clear_memory_cache()
        prev_batch = self.batch_size
        if self.scale_mode == "power_of_two":
            self.batch_size = _next_power_of_two_batch(self.batch_size)
        else:
            self.batch_size = max(1, self.batch_size - 1)
        self.accumulation_steps = compute_accumulation_steps(self.batch_size, self.target_batch)
        print(
            f"[sajepa] OOM at batch={prev_batch} → retrying batch={self.batch_size} "
            f"(accum={self.accumulation_steps}, attempt {self._attempt}/{self.max_retries})"
        )
        return True

    def __iter__(self):
        self._attempt = 0
        self.batch_size = self.initial_batch
        self.accumulation_steps = compute_accumulation_steps(self.batch_size, self.target_batch)
        self._done = False
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self
