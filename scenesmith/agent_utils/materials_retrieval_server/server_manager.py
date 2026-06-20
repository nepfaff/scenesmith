"""Server manager for materials retrieval server lifecycle management."""

import logging
import threading
import time

from threading import Thread

import requests

from omegaconf import DictConfig
from werkzeug.serving import make_server

from scenesmith.agent_utils.materials_retrieval_server.config import MaterialsConfig
from scenesmith.utils.network_utils import is_port_available

from .server_app import MaterialsRetrievalApp

console_logger = logging.getLogger(__name__)


class MaterialsRetrievalServer:
    """Manages the lifecycle of a materials retrieval server.

    The server runs Flask in a separate thread within the same process,
    which avoids the CUDA fork issue that occurs when using multiprocessing with
    CLIP models.

    This class is designed for programmatic usage within experiments or
    applications. For standalone usage (e.g., testing, debugging, or
    microservice deployment), use the standalone_server.py script instead.

    Example:
        >>> server = MaterialsRetrievalServer(host="127.0.0.1", port=7018)
        >>> server.start()
        >>> server.wait_until_ready()
        >>> # ... use server via MaterialsRetrievalClient ...
        >>> server.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7008,
        preload_retriever: bool = True,
        materials_config: MaterialsConfig | DictConfig | None = None,
        clip_device: str | None = None,
    ) -> None:
        """Initialize the materials retrieval server manager.

        Args:
            host: The host address to bind the server to.
            port: The port number to bind to (default: 7008).
            preload_retriever: Whether to preload the materials retriever (includes
                CLIP model loading) on server start. When True, the retriever is loaded
                during initialization, eliminating first-request latency. When False,
                the retriever is loaded lazily on first request. Default: True.
            materials_config: Configuration for materials retrieval. Can be
                MaterialsConfig or DictConfig from Hydra. If None, uses default
                configuration from project defaults.
            clip_device: Target device for CLIP model (e.g., "cuda:0"). If None,
                uses default (cuda if available, else cpu).

        Raises:
            ValueError: If the specified port is not available.
        """
        if not is_port_available(host, port):
            raise ValueError(f"Port {port} is not available on {host}")

        self._host = host
        self._port = port
        self._preload_retriever = preload_retriever
        self._materials_config = materials_config
        self._clip_device = clip_device
        self._app: MaterialsRetrievalApp | None = None
        self._http_server = None
        self._server_thread: Thread | None = None
        self._running = False
        self._shutdown_event = threading.Event()

        console_logger.debug(
            f"Initialized MaterialsRetrievalServer(host={host}, port={port}, "
            f"preload_retriever={preload_retriever}, clip_device={clip_device})"
        )

    def start(self) -> None:
        """Start the materials retrieval server.

        Raises:
            RuntimeError: If server is already running.
        """
        if self._running:
            raise RuntimeError("Server is already running")

        console_logger.info(
            f"Starting materials retrieval server on {self._host}:{self._port}"
        )

        try:
            # Create the Flask application.
            self._app = MaterialsRetrievalApp(
                preload_retriever=self._preload_retriever,
                materials_config=self._materials_config,
                clip_device=self._clip_device,
            )

            # Start the processing queue.
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
                f"Materials retrieval server ready on {self._host}:{self._port}"
            )
            console_logger.info(
                f"Health check URL: http://{self._host}:{self._port}/health"
            )

        except Exception as e:
            self._cleanup()
            console_logger.error(f"Failed to start server: {e}")
            raise

    def stop(self) -> None:
        """Stop the materials retrieval server gracefully."""
        if not self._running:
            console_logger.warning("Server is not running")
            return

        console_logger.info("Stopping materials retrieval server...")

        # Signal shutdown.
        self._shutdown_event.set()

        # Stop the processing queue.
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
        console_logger.info("Materials retrieval server stopped")

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
