"""Fault-tolerant parallel execution utilities.

Provides process isolation for parallel task execution, ensuring that one task
crashing does not affect others. This is critical for long-running batch jobs
where ProcessPoolExecutor's "broken pool" behavior is problematic.
"""

import logging
import multiprocessing
import queue
import signal
import traceback

from multiprocessing.connection import wait
from typing import Any, Callable

console_logger = logging.getLogger(__name__)


def _get_signal_name(exit_code: int) -> str:
    """Get human-readable signal name from exit code.

    Negative exit codes indicate the process was killed by a signal.
    For example, -11 means SIGSEGV (segmentation fault).

    Args:
        exit_code: Process exit code (negative for signals).

    Returns:
        Signal name if exit_code is negative, otherwise empty string.
    """
    if exit_code >= 0:
        return ""
    signal_num = -exit_code
    try:
        return f" ({signal.Signals(signal_num).name})"
    except (ValueError, AttributeError):
        return ""


def _reset_worker_logging() -> None:
    """Reset logging handlers at start of worker process.

    Prevents file descriptor inheritance issues if a platform or configured
    multiprocessing context inherits handlers from the parent process.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()


def _worker_wrapper(
    target: Callable,
    kwargs: dict,
    task_id: str,
    result_queue: multiprocessing.Queue,
    return_values: bool,
) -> None:
    """Wrapper that runs target function and reports result to queue."""
    _reset_worker_logging()
    console_logger.debug(f"Worker {task_id} starting: {target.__name__}")
    try:
        result = target(**kwargs)
        if return_values:
            result_queue.put((task_id, True, result))
        else:
            result_queue.put((task_id, True, None))
    except Exception as e:
        # Preserve full traceback for debugging, not just str(e).
        error_msg = f"{e}\n{traceback.format_exc()}"
        console_logger.error(f"Worker {task_id} failed: {error_msg}")
        result_queue.put((task_id, False, error_msg))


def run_parallel_isolated(
    tasks: list[tuple[str, Callable, dict]],
    max_workers: int,
    return_values: bool = False,
) -> dict[str, tuple[bool, Any]]:
    """Run tasks in isolated processes with fault tolerance.

    Spawns up to max_workers processes at a time. As each completes, spawns the
    next task. One process crashing does not affect others.

    Unlike ProcessPoolExecutor, this function:
    - Spawns a fresh process per task (clean state, no accumulated resources)
    - Continues running other tasks if one crashes
    - Uses efficient wait() instead of polling

    Args:
        tasks: List of (task_id, target_function, kwargs) tuples. The target
            function will be called with **kwargs.
        max_workers: Maximum number of concurrent processes.
        return_values: If True, capture and return values from target functions.
            If False, only track success/failure status.

    Returns:
        Dict mapping task_id to (success: bool, result_or_error).
        For successful tasks: result_or_error is the return value (if
        return_values=True) or None (if return_values=False).
        For failed tasks: result_or_error is the error message string.
    """
    mp_ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = mp_ctx.Queue()
    pending = list(tasks)
    active: dict[int, tuple[multiprocessing.Process, str]] = {}
    results: dict[str, tuple[bool, Any]] = {}

    while pending or active:
        # Spawn processes up to max_workers.
        while len(active) < max_workers and pending:
            task_id, target, kwargs = pending.pop(0)
            p = mp_ctx.Process(
                target=_worker_wrapper,
                args=(target, kwargs, task_id, result_queue, return_values),
            )
            p.start()
            active[p.pid] = (p, task_id)
            console_logger.info(f"Started {task_id} (pid={p.pid})")

        # Wait for any process to finish (efficient, no busy polling).
        if active:
            sentinels = [proc.sentinel for proc, _ in active.values()]
            wait(sentinels, timeout=1.0)

        # Drain all available results from queue first. This avoids race
        # conditions when multiple processes finish simultaneously.
        while True:
            try:
                result_task_id, success, result_or_error = result_queue.get_nowait()
                results[result_task_id] = (success, result_or_error)
                status = "completed" if success else f"failed: {result_or_error}"
                console_logger.info(f"{result_task_id} {status}")
            except queue.Empty:
                break

        # Collect finished processes. Any that didn't report results crashed.
        for pid, (proc, task_id) in list(active.items()):
            if not proc.is_alive():
                proc.join()
                del active[pid]

                # If task didn't report via queue, it crashed (e.g., SIGKILL, OOM).
                if task_id not in results:
                    signal_name = _get_signal_name(proc.exitcode)
                    results[task_id] = (
                        False,
                        f"Process crashed (exitcode={proc.exitcode}{signal_name})",
                    )
                    console_logger.error(
                        f"{task_id} crashed (exitcode={proc.exitcode}{signal_name})"
                    )

    return results
