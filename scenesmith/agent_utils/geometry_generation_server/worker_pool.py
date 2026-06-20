"""GPU worker pool for multi-GPU geometry generation.

This module manages a pool of GPU worker processes, distributing geometry
generation requests across all available GPUs for parallel processing.

Key design principles:
1. Workers are spawned BEFORE any CUDA initialization in the parent process
2. Each worker sets CUDA_VISIBLE_DEVICES to its assigned GPU index
3. On-demand dispatch blocks until a worker is available (preserves fair scheduling)
4. Single code path works for both single-GPU and multi-GPU configurations
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
import uuid

from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from threading import Thread
from typing import Callable

from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerRequest,
)
from scenesmith.agent_utils.geometry_generation_server.gpu_worker import (
    ShutdownRequest,
    WorkerReady,
    WorkRequest,
    WorkResult,
    gpu_worker_main,
)

console_logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Information about a worker process."""

    gpu_id: int
    """The GPU index this worker is assigned to."""

    process: Process
    """The worker process."""

    work_queue: Queue
    """Queue to send work to this worker."""


@dataclass
class PoolStats:
    """Statistics from the worker pool for health reporting."""

    num_workers: int
    """Number of GPU workers in the pool."""

    total_requests: int
    """Total number of requests processed."""

    completed_requests: int
    """Number of successfully completed requests."""

    failed_requests: int
    """Number of failed requests."""

    avg_processing_time_s: float | None
    """Average processing time (GPU only) in seconds, or None if no data."""

    avg_end_to_end_latency_s: float | None
    """Average end-to-end latency in seconds, or None if no data."""

    avg_queue_wait_s: float | None
    """Average queue wait time (end-to-end - processing), or None if no data."""

    max_queue_wait_s: float | None
    """Maximum queue wait time observed over server lifecycle, or None if no data."""

    worker_details: list[dict]
    """Per-worker statistics."""


class GPUWorkerPool:
    """Manages pool of GPU worker processes with on-demand dispatch.

    Workers signal availability after completing each request.
    submit_request() blocks until a worker is free, ensuring:
    - Fair scheduler ordering is preserved (requests dispatched in order)
    - Natural load balancing (faster GPUs process more requests)
    - Works identically with 1 GPU or N GPUs (single code path)

    Example:
        >>> pool = GPUWorkerPool(use_mini=False, backend="hunyuan3d")
        >>> pool.start()
        >>> print(f"Pool has {pool.num_workers} workers")
        >>>
        >>> def callback(index, result):
        ...     print(f"Request {index}: {result}")
        >>>
        >>> pool.submit_request(request, callback, request_index=0)
        >>> pool.stop()
    """

    def __init__(
        self,
        use_mini: bool = False,
        backend: str = "hunyuan3d",
        sam3d_config: dict | None = None,
        preload_pipeline: bool = True,
        log_file: Path | None = None,
    ) -> None:
        """Initialize the GPU worker pool.

        Args:
            use_mini: Whether to use mini model variant (Hunyuan3D only).
            backend: Generation backend ("hunyuan3d" or "sam3d").
            sam3d_config: Configuration for SAM3D backend.
            preload_pipeline: Whether to preload pipeline in workers on start.
            log_file: Optional path to log file for worker logging.
        """
        self._use_mini = use_mini
        self._backend = backend
        self._sam3d_config = sam3d_config
        self._preload_pipeline = preload_pipeline
        self._log_file = str(log_file) if log_file else None

        # Detect available GPUs (respects CUDA_VISIBLE_DEVICES if set).
        self._gpu_ids = self._detect_gpu_ids()
        self._num_gpus = len(self._gpu_ids)
        console_logger.info(
            f"Detected {self._num_gpus} GPU(s) for worker pool: {self._gpu_ids}"
        )

        # CUDA is not fork-safe once any parent process has initialized it. Use
        # spawn so geometry workers always start from a clean interpreter before
        # setting CUDA_VISIBLE_DEVICES and importing torch/CUDA code.
        self._mp_ctx = mp.get_context("spawn")

        # Lock for serializing pipeline initialization to avoid I/O contention.
        # SAM3D checkpoints are ~15GB total. Loading them on 8 workers simultaneously
        # causes severe disk I/O contention. This lock ensures only one worker loads
        # checkpoints at a time.
        self._init_lock = self._mp_ctx.Lock()

        # Worker tracking.
        self._workers: dict[int, WorkerInfo] = {}
        self._available_workers: Queue = self._mp_ctx.Queue()
        self._result_queue: Queue = self._mp_ctx.Queue()
        self._pending_callbacks: dict[str, tuple[Callable, int]] = {}
        self._pending_callbacks_lock = threading.Lock()

        # Result collector thread.
        self._result_thread: Thread | None = None
        self._health_monitor_thread: Thread | None = None
        self._running = False

        # Aggregate statistics.
        self._total_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._processing_times: list[float] = []
        self._end_to_end_latencies: list[float] = []
        self._max_queue_wait: float | None = None
        self._stats_lock = threading.Lock()

        # Per-worker statistics for utilization tracking.
        self._per_worker_completed: dict[int, int] = {}
        self._per_worker_failed: dict[int, int] = {}

    @property
    def num_workers(self) -> int:
        """Get the number of workers in the pool."""
        return self._num_gpus

    def _start_single_worker(self, gpu_id: int) -> None:
        """Start a single GPU worker process.

        Args:
            gpu_id: The GPU index to start the worker on.
        """
        work_queue = self._mp_ctx.Queue()

        process = self._mp_ctx.Process(
            target=gpu_worker_main,
            kwargs={
                "gpu_id": gpu_id,
                "work_queue": work_queue,
                "result_queue": self._result_queue,
                "use_mini": self._use_mini,
                "backend": self._backend,
                "sam3d_config": self._sam3d_config,
                "preload_pipeline": self._preload_pipeline,
                "init_lock": self._init_lock,
                "log_file": self._log_file,
            },
        )
        process.start()

        self._workers[gpu_id] = WorkerInfo(
            gpu_id=gpu_id, process=process, work_queue=work_queue
        )

        console_logger.info(f"Started worker for GPU {gpu_id} (PID: {process.pid})")

    def _restart_worker(self, gpu_id: int) -> None:
        """Restart a dead worker.

        Args:
            gpu_id: The GPU index of the worker to restart.
        """
        old_worker = self._workers.get(gpu_id)
        if old_worker:
            # Clean up old process if still somehow alive.
            if old_worker.process.is_alive():
                old_worker.process.terminate()
                old_worker.process.join(timeout=5.0)

        # Start new worker.
        self._start_single_worker(gpu_id)
        console_logger.info(f"Restarted worker {gpu_id}")

    def _health_monitor_loop(self) -> None:
        """Monitor worker health and restart dead workers."""
        while self._running:
            time.sleep(5.0)

            # Re-check after sleep to avoid restarting during shutdown.
            if not self._running:
                break

            for gpu_id, worker in list(self._workers.items()):
                if not self._running:
                    break
                if not worker.process.is_alive():
                    console_logger.warning(
                        f"Worker {gpu_id} (PID {worker.process.pid}) died, restarting..."
                    )
                    self._restart_worker(gpu_id)

    def start(self) -> None:
        """Start all GPU worker processes.

        Uses 'spawn' context to create clean worker processes. Each worker sets
        CUDA_VISIBLE_DEVICES before importing CUDA-dependent code.

        Workers are staggered to avoid contention during pipeline loading.
        """
        if self._running:
            raise RuntimeError("Worker pool is already running")

        console_logger.info(
            f"Starting GPU worker pool with {self._num_gpus} workers..."
        )
        self._running = True

        # Spawn worker processes on all available GPUs.
        for i, gpu_id in enumerate(self._gpu_ids):
            self._start_single_worker(gpu_id)

            # Stagger worker starts to avoid contention during pipeline loading.
            if i < self._num_gpus - 1 and self._preload_pipeline:
                time.sleep(1.0)

        # Start result collector thread.
        self._result_thread = Thread(target=self._collect_results, daemon=False)
        self._result_thread.start()

        # Start health monitor thread.
        self._health_monitor_thread = Thread(
            target=self._health_monitor_loop, daemon=True, name="WorkerHealthMonitor"
        )
        self._health_monitor_thread.start()

        console_logger.info("GPU worker pool started successfully")

    def stop(self) -> None:
        """Stop all worker processes gracefully."""
        if not self._running:
            console_logger.warning("Worker pool is not running")
            return

        console_logger.info("Stopping GPU worker pool...")
        self._running = False

        # Send shutdown signal to all workers.
        for gpu_id, worker in self._workers.items():
            try:
                worker.work_queue.put(ShutdownRequest())
                console_logger.debug(f"Sent shutdown signal to worker {gpu_id}")
            except Exception as e:
                console_logger.warning(
                    f"Failed to send shutdown to worker {gpu_id}: {e}"
                )

        # Wait for health monitor to stop (daemon, checks _running flag).
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=2.0)

        # Wait for result collector to finish.
        if self._result_thread and self._result_thread.is_alive():
            self._result_thread.join(timeout=5)
            if self._result_thread.is_alive():
                console_logger.warning(
                    "Result collector thread did not stop gracefully"
                )

        # Wait for workers to finish (with timeout).
        for gpu_id, worker in self._workers.items():
            if worker.process.is_alive():
                worker.process.join(timeout=10)
                if worker.process.is_alive():
                    console_logger.warning(
                        f"Worker {gpu_id} did not stop gracefully, terminating..."
                    )
                    worker.process.terminate()
                    worker.process.join(timeout=2)

        # Clean up.
        self._workers.clear()
        console_logger.info("GPU worker pool stopped")

    def submit_request(
        self,
        request: GeometryGenerationServerRequest,
        callback: Callable[[int, tuple[str, dict | str]], None],
        request_index: int,
        received_timestamp: float,
    ) -> None:
        """Submit a request to an available worker.

        This method blocks until a worker is available, preserving the fair
        ordering from the StrictRoundRobinScheduler.

        Args:
            request: The geometry generation request.
            callback: Function to call with (index, result) when complete.
            request_index: Index of this request in the batch.
            received_timestamp: Time when request was received by server.
        """
        if not self._running:
            raise RuntimeError("Worker pool is not running")

        # Block until a worker is available.
        worker_id = self._available_workers.get()

        # Generate unique request ID.
        request_id = str(uuid.uuid4())

        # Store callback for later invocation.
        with self._pending_callbacks_lock:
            self._pending_callbacks[request_id] = (callback, request_index)

        # Track request.
        with self._stats_lock:
            self._total_requests += 1

        # Submit work to the worker.
        worker = self._workers[worker_id]
        worker.work_queue.put(
            WorkRequest(
                request_id=request_id,
                request=request,
                received_timestamp=received_timestamp,
            )
        )

        console_logger.info(
            f"Submitted request {request_id} to worker {worker_id}: {request.prompt}"
        )

    def get_stats(self) -> PoolStats:
        """Get aggregate statistics from the pool.

        Returns:
            Pool statistics for health reporting.
        """
        with self._stats_lock:
            avg_time = None
            if self._processing_times:
                avg_time = sum(self._processing_times) / len(self._processing_times)

            avg_latency = None
            if self._end_to_end_latencies:
                avg_latency = sum(self._end_to_end_latencies) / len(
                    self._end_to_end_latencies
                )

            # Compute queue wait time as difference between latency and processing.
            avg_queue_wait = None
            if avg_latency is not None and avg_time is not None:
                avg_queue_wait = avg_latency - avg_time

            # Build per-worker details with utilization stats.
            total_processed = self._completed_requests + self._failed_requests
            worker_details = []
            for gpu_id, worker in self._workers.items():
                completed = self._per_worker_completed.get(gpu_id, 0)
                failed = self._per_worker_failed.get(gpu_id, 0)
                worker_total = completed + failed
                proportion = (
                    worker_total / total_processed if total_processed > 0 else 0
                )

                worker_details.append(
                    {
                        "gpu_id": gpu_id,
                        "pid": worker.process.pid,
                        "alive": worker.process.is_alive(),
                        "completed_requests": completed,
                        "failed_requests": failed,
                        "total_requests": worker_total,
                        "proportion": round(proportion, 4),
                    }
                )

            return PoolStats(
                num_workers=self._num_gpus,
                total_requests=self._total_requests,
                completed_requests=self._completed_requests,
                failed_requests=self._failed_requests,
                avg_processing_time_s=avg_time,
                avg_end_to_end_latency_s=avg_latency,
                avg_queue_wait_s=avg_queue_wait,
                max_queue_wait_s=self._max_queue_wait,
                worker_details=worker_details,
            )

    def _collect_results(self) -> None:
        """Collect results from workers and invoke callbacks.

        This runs in a separate thread, continuously collecting results from
        the shared result queue and routing them back to the appropriate
        callbacks. Also handles WorkerReady signals from workers that have
        finished initialization.
        """
        console_logger.debug("Result collector thread started")

        while self._running or not self._result_queue.empty():
            try:
                message = self._result_queue.get(timeout=0.5)
            except Empty:
                continue

            # Handle worker ready signal (worker finished initialization).
            if isinstance(message, WorkerReady):
                console_logger.info(
                    f"Worker {message.worker_id} initialized and ready for requests"
                )
                self._available_workers.put(message.worker_id)
                continue

            # Handle work result.
            result: WorkResult = message

            # Look up and invoke callback.
            with self._pending_callbacks_lock:
                callback_info = self._pending_callbacks.pop(result.request_id, None)

            if callback_info is None:
                console_logger.warning(
                    f"No callback found for request {result.request_id}"
                )
                # Still return worker to available pool.
                self._available_workers.put(result.worker_id)
                continue

            callback, request_index = callback_info

            # Update statistics (aggregate and per-worker).
            with self._stats_lock:
                if result.status == "success":
                    self._completed_requests += 1
                    self._per_worker_completed[result.worker_id] = (
                        self._per_worker_completed.get(result.worker_id, 0) + 1
                    )
                else:
                    self._failed_requests += 1
                    self._per_worker_failed[result.worker_id] = (
                        self._per_worker_failed.get(result.worker_id, 0) + 1
                    )

                # Aggregate processing time from worker.
                if result.processing_time_seconds is not None:
                    self._processing_times.append(result.processing_time_seconds)
                    # Keep only last 10000 times to bound memory.
                    if len(self._processing_times) > 10000:
                        self._processing_times.pop(0)

                # Aggregate end-to-end latency from worker.
                if result.end_to_end_latency_seconds is not None:
                    self._end_to_end_latencies.append(result.end_to_end_latency_seconds)
                    # Keep only last 10000 latencies to bound memory.
                    if len(self._end_to_end_latencies) > 10000:
                        self._end_to_end_latencies.pop(0)

                # Track max queue wait time when both measurements available.
                if (
                    result.end_to_end_latency_seconds is not None
                    and result.processing_time_seconds is not None
                ):
                    queue_wait = (
                        result.end_to_end_latency_seconds
                        - result.processing_time_seconds
                    )
                    if (
                        self._max_queue_wait is None
                        or queue_wait > self._max_queue_wait
                    ):
                        self._max_queue_wait = queue_wait

            # Invoke callback.
            try:
                if result.status == "success":
                    callback(request_index, ("success", result.data))
                else:
                    callback(request_index, ("error", result.error))
            except Exception as e:
                console_logger.error(
                    f"Callback failed for request {result.request_id}: {e}"
                )

            # Return worker to available pool.
            self._available_workers.put(result.worker_id)

        console_logger.debug("Result collector thread stopped")

    @staticmethod
    def _detect_gpu_ids() -> list[int]:
        """Detect available GPU IDs, respecting CUDA_VISIBLE_DEVICES if set.

        Uses nvidia-smi to avoid importing torch (which would initialize CUDA
        in the parent process).

        If CUDA_VISIBLE_DEVICES is set, only those GPUs are returned.
        Otherwise, all available GPUs are returned.

        Returns:
            List of physical GPU IDs to use.
        """
        # Check if CUDA_VISIBLE_DEVICES is set.
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is not None and cuda_visible.strip():
            # Parse the comma-separated list of GPU IDs.
            try:
                gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
                if gpu_ids:
                    console_logger.info(
                        f"Using GPUs from CUDA_VISIBLE_DEVICES: {gpu_ids}"
                    )
                    return gpu_ids
            except ValueError:
                console_logger.warning(
                    f"Failed to parse CUDA_VISIBLE_DEVICES='{cuda_visible}', "
                    "falling back to nvidia-smi detection"
                )

        # Fall back to detecting all GPUs via nvidia-smi.
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpu_ids = [int(line.strip()) for line in lines if line.strip()]
                if gpu_ids:
                    return gpu_ids
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            console_logger.warning(f"nvidia-smi detection failed: {e}")

        console_logger.warning("Could not detect GPUs, defaulting to GPU 0")
        return [0]
