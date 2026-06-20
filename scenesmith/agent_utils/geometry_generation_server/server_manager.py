"""Geometry generation server lifecycle management.

This module provides the GeometryGenerationServer class for managing the
lifecycle of the geometry generation server. The server automatically
detects and uses all available GPUs.

CRITICAL: This module must NOT import any CUDA-dependent code. The GPU workers
handle all CUDA initialization to enable proper GPU isolation.
"""

import logging
import threading
import time

from pathlib import Path
from threading import Thread

import requests

from werkzeug.serving import make_server

from scenesmith.utils.network_utils import is_port_available

from .server_app import GeometryGenerationApp

console_logger = logging.getLogger(__name__)


class GeometryGenerationServer:
    """Manages the lifecycle of a geometry generation server with multi-GPU support.

    The server automatically detects all available GPUs and spawns one worker
    process per GPU. Use CUDA_VISIBLE_DEVICES to control which GPUs are used.

    This class is designed for programmatic usage within experiments or
    applications. For standalone usage (e.g., testing, debugging, or
    microservice deployment), use the standalone_server.py script instead.

    Example (Hunyuan3D):
        >>> server = GeometryGenerationServer(
        ...     host="127.0.0.1",
        ...     port=7000,
        ...     backend="hunyuan3d"
        ... )
        >>> server.start()
        >>> server.wait_until_ready()
        >>> # ... use server via GeometryGenerationClient ...
        >>> server.stop()

    Example (SAM3D):
        >>> sam3d_config = {
        ...     "sam3_checkpoint": "external/checkpoints/sam3_hiera_b+_1104.pt",
        ...     "sam3d_checkpoint": "external/checkpoints/sam_3d_objects.ckpt",
        ...     "mode": "foreground",
        ...     "text_prompt": None,
        ...     "threshold": 0.5,
        ... }
        >>> server = GeometryGenerationServer(
        ...     host="127.0.0.1",
        ...     port=7000,
        ...     backend="sam3d",
        ...     sam3d_config=sam3d_config
        ... )
        >>> server.start()
        >>> server.wait_until_ready()
        >>> # ... use server via GeometryGenerationClient ...
        >>> server.stop()

    Multi-GPU:
        Multi-GPU mode is automatically enabled when multiple GPUs are detected.
        The server spawns one worker process per visible GPU.

        To control which GPUs are used:
        >>> # Use only GPUs 0 and 1
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        >>> server = GeometryGenerationServer(...)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7000,
        preload_pipeline: bool = True,
        use_mini: bool = False,
        backend: str = "hunyuan3d",
        sam3d_config: dict | None = None,
        log_file: Path | None = None,
    ) -> None:
        """Initialize the geometry generation server manager.

        Args:
            host: The host address to bind the server to.
            port: The port number to bind to.
            preload_pipeline: Whether to preload the pipeline in workers on start.
                When True, the pipeline is loaded during server initialization,
                eliminating first-request latency. When False, the pipeline is loaded
                lazily on first request. Default: True.
            use_mini: Whether to use the mini model variant (0.6B parameters) instead
                of the full model. Only applies to Hunyuan3D backend. The mini model
                is faster with lower memory usage but may have reduced quality.
                Default: False (use full model).
            backend: Which 3D generation backend to use ("hunyuan3d" or "sam3d").
                Default: "hunyuan3d".
            sam3d_config: Configuration for SAM3D backend. Required if backend="sam3d".
                Should contain sam3_checkpoint and sam3d_checkpoint paths.
            log_file: Optional path to log file for worker logging (e.g., experiment.log).

        Raises:
            ValueError: If the specified port is not available or if backend="sam3d"
                but sam3d_config is not provided.
        """
        if not is_port_available(host, port):
            raise ValueError(f"Port {port} is not available on {host}")

        if backend not in ["hunyuan3d", "sam3d"]:
            raise ValueError(f"Unknown backend: {backend}")

        if backend == "sam3d" and sam3d_config is None:
            raise ValueError("sam3d_config is required when backend='sam3d'")

        self._host = host
        self._port = port
        self._preload_pipeline = preload_pipeline
        self._use_mini = use_mini
        self._backend = backend
        self._sam3d_config = sam3d_config
        self._log_file = log_file
        self._app: GeometryGenerationApp | None = None
        self._http_server = None
        self._server_thread: Thread | None = None
        self._running = False
        self._shutdown_event = threading.Event()

        console_logger.debug(
            f"Initialized GeometryGenerationServer(host={host}, port={port}, "
            f"preload_pipeline={preload_pipeline}, use_mini={use_mini}, "
            f"backend={backend})"
        )

    def start(self) -> None:
        """Start the geometry generation server.

        This spawns GPU worker processes (one per available GPU) and starts
        the Flask HTTP server. Worker processes preload the pipeline if
        preload_pipeline=True was specified.

        Raises:
            RuntimeError: If server is already running.
        """
        if self._running:
            raise RuntimeError("Server is already running")

        console_logger.info(
            f"Starting geometry generation server on {self._host}:{self._port}"
        )

        try:
            # Create the Flask application with all parameters.
            # Workers are created but not started yet.
            self._app = GeometryGenerationApp(
                use_mini=self._use_mini,
                backend=self._backend,
                sam3d_config=self._sam3d_config,
                preload_pipeline=self._preload_pipeline,
                log_file=self._log_file,
            )

            # Start the worker pool and coordinator thread.
            # This spawns GPU workers and preloads pipelines if enabled.
            self._app.start_processing()

            # Start Flask server in a separate thread.
            self._server_thread = Thread(
                target=self._run_server,
                daemon=False,  # Not daemon so we can shut down cleanly.
            )
            self._server_thread.start()

            # Wait for the server to be ready.
            self._wait_until_ready()
            self._running = True

            console_logger.info(
                f"Geometry generation server ready on {self._host}:{self._port}"
            )
            console_logger.info(
                f"Health check URL: http://{self._host}:{self._port}/health"
            )

        except Exception as e:
            self._cleanup()
            console_logger.error(f"Failed to start server: {e}")
            raise

    def stop(self) -> None:
        """Stop the geometry generation server gracefully."""
        if not self._running:
            console_logger.warning("Server is not running")
            return

        console_logger.info("Stopping geometry generation server...")

        # Signal shutdown.
        self._shutdown_event.set()

        # Stop the worker pool and coordinator thread.
        if self._app:
            self._app.stop_processing()

        if self._http_server is not None:
            self._http_server.shutdown()

        # Wait for server thread to complete.
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
            if self._server_thread.is_alive():
                console_logger.warning("Server thread did not stop gracefully")

        self._cleanup()
        console_logger.info("Geometry generation server stopped")

    def wait_until_ready(self, timeout_s: float = 30) -> None:
        """Wait for the server to be ready to accept requests.

        Args:
            timeout_s: Maximum time to wait for server readiness.

        Raises:
            RuntimeError: If server doesn't become ready within timeout.
        """
        if not self._running:
            raise RuntimeError("Server is not running")

        self._wait_until_ready(timeout_s)

    def is_running(self) -> bool:
        """Check if the server is currently running.

        Returns:
            True if server is running and ready.
        """
        return self._running

    @property
    def host(self) -> str:
        """Get the server host address."""
        return self._host

    @property
    def port(self) -> int:
        """Get the server port number."""
        return self._port

    def _run_server(self) -> None:
        """Run the Flask server in a separate thread."""
        try:
            self._http_server = make_server(
                self._host, self._port, self._app, threaded=True
            )
            self._http_server.serve_forever()
        except Exception as e:
            if not self._shutdown_event.is_set():
                console_logger.error(f"Server thread failed: {e}")
            self._shutdown_event.set()
        finally:
            if self._http_server is not None:
                self._http_server.server_close()

    def _wait_until_ready(self, timeout: float = 30) -> None:
        """Wait for server to be ready to accept requests.

        Args:
            timeout: Maximum time to wait.

        Raises:
            RuntimeError: If server doesn't become ready within timeout.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://{self._host}:{self._port}/health", timeout=1
                )
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass

            time.sleep(0.1)

        raise RuntimeError(f"Server did not become ready within {timeout} seconds")

    def _cleanup(self) -> None:
        """Clean up server resources."""
        self._running = False
        self._app = None
        self._http_server = None
        self._server_thread = None
        self._shutdown_event.clear()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
