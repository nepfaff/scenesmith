import contextlib
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

from pathlib import Path

import requests

from scenesmith.agent_utils.mesh_utils import convert_gltf_to_glb
from scenesmith.utils.network_utils import find_available_port, is_port_available

console_logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None


class BlenderServer:
    """Manages a Flask-based Blender rendering server.

    The server runs in a separate process because bpy (Blender Python API) is
    not thread-safe and must run in the main thread. By using subprocess, we
    ensure bpy runs in the main thread of the server process.

    The server provides endpoints for:
    - Standard scene rendering (/render)
    - Metric overlay rendering (/render_metric)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        port_range: tuple[int, int] | None = None,
        blend_file: Path | None = None,
        bpy_settings_file: Path | None = None,
        server_startup_delay: float = 3.0,
        port_cleanup_delay: float = 2.0,
        gpu_id: int | None = None,
        log_file: Path | None = None,
    ) -> None:
        """Initialize the Blender server manager.

        Args:
            host: The host address to bind the server to.
            port: The specific port number to bind to. If None, will use port_range.
            port_range: Tuple of (start_port, end_port) to search for available
                port. Defaults to (8000, 8050) if neither port nor port_range
                specified.
            blend_file: Optional path to a .blend file to use as base scene.
            bpy_settings_file: Optional path to a .py file with Blender settings.
            server_startup_delay: Seconds to wait after starting server subprocess
                                 to allow Blender initialization.
            port_cleanup_delay: Seconds to wait after stopping server to allow
                               OS port cleanup.
            gpu_id: Optional GPU device ID to restrict this server to. When set,
                   uses bubblewrap to isolate the server process to only see
                   the specified GPU. This enables distributing Blender rendering
                   across multiple GPUs for parallel scene generation.
            log_file: Optional path to capture server stdout/stderr. If None,
                     output inherits from parent process. Useful for debugging
                     server crashes.

        Raises:
            ValueError: If blend_file or bpy_settings_file paths don't exist,
                       or if both port and port_range are specified.
        """
        # Validate file paths if provided.
        if blend_file is not None and not blend_file.exists():
            raise ValueError(f"Blend file not found: {blend_file}")
        if bpy_settings_file is not None and not bpy_settings_file.exists():
            raise ValueError(f"Bpy settings file not found: {bpy_settings_file}")

        # Validate port configuration.
        if port is not None and port_range is not None:
            raise ValueError("Cannot specify both port and port_range")

        # Set default port range if neither port nor port_range specified.
        if port is None and port_range is None:
            port_range = (8000, 8050)

        self._host = host
        self._port = port
        self._port_range = port_range
        self._actual_port: int | None = None
        self._blend_file = blend_file
        self._bpy_settings_file = bpy_settings_file
        self._server_startup_delay = server_startup_delay
        self._port_cleanup_delay = port_cleanup_delay
        self._gpu_id = gpu_id
        self._log_file = log_file
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._server_process: subprocess.Popen | None = None
        self._running = False

        console_logger.debug(
            f"Initialized BlenderServer(host={host}, port={port}, "
            f"port_range={port_range}, blend_file={blend_file}, "
            f"bpy_settings_file={bpy_settings_file}, "
            f"server_startup_delay={server_startup_delay}, "
            f"port_cleanup_delay={port_cleanup_delay}, gpu_id={gpu_id}, "
            f"log_file={log_file})"
        )

    def start(self) -> None:
        """Start the Blender server in a separate process.

        Raises:
            RuntimeError: If server is already running, or if no port is available.
            FileNotFoundError: If standalone server script not found.
            subprocess.SubprocessError: If server process fails to start.
        """
        if self._running:
            raise RuntimeError("Server is already running")

        try:
            with self._port_allocation_lock():
                self._start_process_with_allocated_port()
        except Exception as e:
            # Clean up on failure.
            self._cleanup_startup_failure()
            console_logger.error(f"Failed to start server: {e}")
            raise

    def _start_process_with_allocated_port(self) -> None:
        """Start the subprocess while holding any needed port-allocation lock."""
        # Determine the port to use.
        target_port = self._determine_port()
        self._actual_port = target_port
        console_logger.info(f"Starting Blender server on {self._host}:{target_port}")

        try:
            # Create a temporary directory for the server.
            self._temp_dir = tempfile.TemporaryDirectory(prefix="blender_server_")
            console_logger.debug(f"Created temp directory: {self._temp_dir.name}")

            # Build command to start the server process using standalone script.
            standalone_script = Path(__file__).parent / "standalone_server.py"
            if not standalone_script.exists():
                raise FileNotFoundError(
                    f"Standalone server script not found: {standalone_script}"
                )

            project_root = Path(__file__).parent.parent.parent.parent
            server_python = os.environ.get("SCENESMITH_BLENDER_PYTHON") or None
            if server_python is None:
                server_python = self._find_blender_python(project_root)
            if server_python is None:
                raise RuntimeError(
                    "Blender Python module 'bpy' is not installed in the current "
                    "environment and no .venv-blender interpreter was found. Run "
                    "scripts/setup_blender_python.sh or set SCENESMITH_BLENDER_PYTHON "
                    "to a Python executable that can import bpy."
                )

            cmd = [
                server_python,
                str(standalone_script),
                "--host",
                self._host,
                "--port",
                str(target_port),
                "--temp-dir",
                self._temp_dir.name,
            ]

            if self._blend_file:
                cmd.extend(["--blend-file", str(self._blend_file)])
                console_logger.debug(f"Using blend file: {self._blend_file}")
            if self._bpy_settings_file:
                cmd.extend(["--bpy-settings-file", str(self._bpy_settings_file)])
                console_logger.debug(
                    f"Using bpy settings file: {self._bpy_settings_file}"
                )
            if self._log_file:
                cmd.extend(["--log-file", str(self._log_file)])
                console_logger.info(f"BlenderServer logs → {self._log_file}")

            console_logger.debug(f"Server command: {' '.join(cmd)}")

            # Set PYTHONPATH so subprocess can find scenesmith module.
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")

            # Apply GPU isolation via bubblewrap if gpu_id is set.
            if self._gpu_id is not None:
                if self._is_bwrap_available():
                    cmd = self._build_bwrap_command(cmd=cmd, gpu_id=self._gpu_id)
                    console_logger.info(
                        f"BlenderServer using GPU {self._gpu_id} via bubblewrap"
                    )
                else:
                    console_logger.warning(
                        f"GPU isolation requested (gpu_id={self._gpu_id}) but "
                        "bubblewrap not installed. Install with: "
                        "sudo apt-get install bubblewrap. "
                        "Falling back to shared GPU access."
                    )

            # Start the server process.
            self._server_process = subprocess.Popen(
                cmd,
                text=True,
                env=env,
                cwd=project_root,
            )

            # Wait for Blender to complete initialization.
            time.sleep(self._server_startup_delay)
            self._raise_if_server_exited_during_startup(target_port=target_port)
            if self._port_range is not None:
                self._wait_until_port_bound(target_port=target_port)

            self._running = True
            console_logger.info(
                f"Server process started with PID {self._server_process.pid}"
            )

        except Exception:
            self._cleanup_startup_failure()
            raise

    @contextlib.contextmanager
    def _port_allocation_lock(self):
        """Serialize port selection for range-based Blender servers.

        Parallel scene workers can otherwise both observe the same free port before
        either Flask process has bound it. Holding this lock until the child binds
        prevents one worker from accidentally using another worker's server.
        """
        if self._port_range is None or fcntl is None:
            yield
            return

        start_port, end_port = self._port_range
        safe_host = self._host.replace("/", "_").replace(":", "_")
        lock_path = (
            Path(tempfile.gettempdir())
            / f"scenesmith_blender_{safe_host}_{start_port}_{end_port}.lock"
        )

        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _raise_if_server_exited_during_startup(self, target_port: int) -> None:
        """Raise if the subprocess exited before startup completed."""
        if self._server_process is None:
            raise RuntimeError("Blender server process was not created")

        exit_code = self._server_process.poll()
        if exit_code is not None:
            raise RuntimeError(
                "Blender server exited during startup "
                f"(exitcode={exit_code}, port={target_port})"
            )

    def _wait_until_port_bound(self, target_port: int, timeout: float = 30.0) -> None:
        """Wait until the child process binds its selected port."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._raise_if_server_exited_during_startup(target_port=target_port)
            if not is_port_available(host=self._host, port=target_port):
                return
            time.sleep(0.1)

        raise RuntimeError(
            f"Blender server did not bind {self._host}:{target_port} "
            f"within {timeout:.1f}s"
        )

    def _cleanup_startup_failure(self) -> None:
        """Clean up resources left by a failed start attempt."""
        if self._server_process is not None:
            if self._server_process.poll() is None:
                self._server_process.terminate()
                try:
                    self._server_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._server_process.kill()
                    self._server_process.wait()
            self._server_process = None
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None
        self._running = False
        self._actual_port = None

    def _determine_port(self) -> int:
        """Determine the port to use for the server."""
        if self._port is not None:
            # Use specific port.
            target_port = self._port
            if not is_port_available(host=self._host, port=target_port):
                raise RuntimeError(
                    f"Port {target_port} is not available on {self._host}"
                )
        else:
            # Find available port in range.
            target_port = find_available_port(
                host=self._host, port_range=self._port_range
            )
            if target_port is None:
                raise RuntimeError(
                    f"No available ports found in range {self._port_range} "
                    f"on {self._host}"
                )
            console_logger.info(
                f"Found available port {target_port} in range {self._port_range}"
            )
        return target_port

    def _is_bwrap_available(self) -> bool:
        """Check if bubblewrap is installed."""
        return shutil.which("bwrap") is not None

    @staticmethod
    def _find_blender_python(project_root: Path) -> str | None:
        """Find a Python executable that can import bpy."""
        candidates: list[Path] = []
        try:
            current_env_has_bpy = importlib.util.find_spec("bpy") is not None
        except (ImportError, ValueError):
            current_env_has_bpy = False

        if current_env_has_bpy:
            candidates.append(Path(sys.executable))

        candidates.extend(
            [
                project_root / ".venv-blender" / "bin" / "python",
                project_root / ".venv-blender" / "Scripts" / "python.exe",
            ]
        )

        for candidate in candidates:
            if candidate.exists() and BlenderServer._python_can_import_bpy(candidate):
                return str(candidate)
        return None

    @staticmethod
    def _python_can_import_bpy(python_executable: Path) -> bool:
        """Check whether an interpreter can import Blender's Python module."""
        try:
            result = subprocess.run(
                [str(python_executable), "-c", "import bpy"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def _build_bwrap_command(self, cmd: list[str], gpu_id: int) -> list[str]:
        """Wrap command in bubblewrap for GPU isolation.

        Uses Linux namespaces to hide all GPU devices except the specified one.
        This enables EEVEE Next (Vulkan) to use only the target GPU, since
        Vulkan ignores CUDA_VISIBLE_DEVICES.

        Args:
            cmd: The command to wrap.
            gpu_id: The GPU device index to expose (e.g., 0 for /dev/nvidia0).

        Returns:
            The wrapped command with bubblewrap prefix.
        """
        # Get home directory for writable bind.
        home_dir = Path.home()
        cwd = Path.cwd()

        bwrap_cmd = [
            "bwrap",
            "--die-with-parent",  # Clean up subprocess when parent exits.
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(home_dir),
            str(home_dir),  # Writable home for outputs.
            "--bind",
            "/tmp",
            "/tmp",  # Shared /tmp for render communication.
            "--bind",
            "/dev/shm",
            "/dev/shm",  # Shared memory for GPU drivers.
            "--proc",
            "/proc",
            "--dev-bind",
            "/dev/urandom",
            "/dev/urandom",
            "--dev-bind",
            "/dev/null",
            "/dev/null",
        ]

        # Bind the working directory writable if it differs from home.
        if cwd != home_dir:
            bwrap_cmd.extend(["--bind", str(cwd), str(cwd)])

        # Re-bind any mount points that sit under the working directory.
        # In Docker, volume mounts (e.g. ./outputs:/app/outputs) are
        # separate mount points that are NOT propagated by --bind /app
        # /app (which only binds the overlay layer). Each volume mount
        # must be explicitly re-bound.
        try:
            cwd_prefix = str(cwd) + "/"
            with open("/proc/mounts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    mount_point = parts[1]
                    if (
                        mount_point.startswith(cwd_prefix)
                        and Path(mount_point).exists()
                    ):
                        bwrap_cmd.extend(["--bind", mount_point, mount_point])
        except OSError:
            pass

        # Bind NVIDIA devices that exist on this system.
        nvidia_devices = [
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            f"/dev/nvidia{gpu_id}",
        ]
        for dev in nvidia_devices:
            if Path(dev).exists():
                bwrap_cmd.extend(["--dev-bind", dev, dev])

        # Bind DRM for Vulkan rendering.
        if Path("/dev/dri").exists():
            bwrap_cmd.extend(["--dev-bind", "/dev/dri", "/dev/dri"])

        bwrap_cmd.append("--")
        return bwrap_cmd + cmd

    def stop(self) -> None:
        """Stop the Blender server and cleanup resources."""
        if not self._running:
            console_logger.debug("Server already stopped")
            return  # Already stopped.

        console_logger.info("Stopping Blender server...")
        self._running = False

        # Terminate the server process.
        if self._server_process:
            pid = self._server_process.pid
            console_logger.debug(f"Terminating server process {pid}")

            self._server_process.terminate()
            try:
                exit_code = self._server_process.wait(timeout=5.0)
                console_logger.debug(
                    f"Server process {pid} exited with code {exit_code}"
                )
            except subprocess.TimeoutExpired:
                console_logger.warning(
                    f"Server process {pid} did not terminate gracefully, killing..."
                )
                self._server_process.kill()
                exit_code = self._server_process.wait()
                console_logger.debug(
                    f"Server process {pid} killed with code {exit_code}"
                )
            finally:
                # Close pipes to prevent resource warnings.
                if self._server_process.stdout:
                    self._server_process.stdout.close()
                if self._server_process.stderr:
                    self._server_process.stderr.close()

            self._server_process = None

        # Clean up temporary directory.
        if self._temp_dir:
            console_logger.debug(f"Cleaning up temp directory: {self._temp_dir.name}")
            self._temp_dir.cleanup()
            self._temp_dir = None

        # Reset the actual port.
        self._actual_port = None

        # Wait for OS to cleanup port.
        time.sleep(self._port_cleanup_delay)

        console_logger.info("Server stopped and cleaned up")

    def is_running(self) -> bool:
        """Check if the server is currently running.

        Returns:
            True if the server is running, False otherwise.
        """
        return self._running

    def get_url(self) -> str:
        """Get the URL where the server is running.

        Returns:
            The server URL.

        Raises:
            RuntimeError: If the server is not running.
        """
        if not self.is_running():
            status = self.get_process_status()
            raise RuntimeError(f"Server is not running (status: {status})")
        return f"http://{self._host}:{self._actual_port}"

    def get_process_status(self) -> str:
        """Get the status of the server process for debugging.

        Returns:
            Human-readable status string.
        """
        if not self._server_process:
            return "No process"

        poll_result = self._server_process.poll()
        if poll_result is None:
            return f"Running (PID {self._server_process.pid})"
        else:
            return f"Exited with code {poll_result}"

    def wait_until_ready(self, timeout: float = 60.0) -> None:
        """Wait until the server is ready to accept HTTP requests.

        Args:
            timeout: Maximum time to wait in seconds.

        Raises:
            RuntimeError: If server is not running or doesn't become ready within
            timeout.
        """
        if not self.is_running():
            raise RuntimeError("Server is not running")

        console_logger.debug(f"Waiting for server to be ready (timeout: {timeout}s)")

        start_time = time.time()
        max_retries = int(timeout)  # Check once per second.

        for i in range(max_retries):
            try:
                response = requests.get(f"{self.get_url()}/", timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    console_logger.debug(f"Server is ready after {elapsed}s")
                    return
            except requests.RequestException as e:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise RuntimeError(
                        f"Server failed to become ready within {timeout}s: {e}"
                    )
                if i < max_retries - 1:
                    time.sleep(1)
                    continue

        raise RuntimeError(f"Server did not become ready within {timeout}s")

    def render_multiview_for_analysis(
        self,
        mesh_path: Path,
        output_dir: Path,
        elevation_degrees: float,
        num_side_views: int = 4,
        include_vertical_views: bool = True,
        width: int = 512,
        height: int = 512,
        timeout: float = 120.0,
        light_energy: float | None = None,
        start_azimuth_degrees: float = 0.0,
        show_coordinate_frame: bool = True,
        taa_samples: int | None = None,
    ) -> list[Path]:
        """Render multiview images for VLM validation via HTTP request.

        If the server crashes during rendering, it will be automatically restarted
        and the request retried.

        Args:
            mesh_path: Path to the mesh file (GLB/GLTF) to render.
            output_dir: Directory where rendered images will be saved.
            elevation_degrees: Elevation angle in degrees for side view cameras.
            num_side_views: Number of equidistant side views (default: 4).
            include_vertical_views: Whether to include top/bottom views.
            width: Image width in pixels.
            height: Image height in pixels.
            timeout: HTTP request timeout in seconds.
            light_energy: Light energy in watts. If None, uses server default.
            start_azimuth_degrees: Starting azimuth angle for side views (default: 0).
                Use 0 for first view at +X, 90 for first view at +Y. Useful for
                wall-mounted objects where front face is at +Y.
            show_coordinate_frame: If True, show RGB coordinate axes overlay.
                Set to False for cleaner validation renders.

        Returns:
            List of paths to rendered PNG images.

        Raises:
            RuntimeError: If server is not running or rendering fails after retries.
        """
        if not self.is_running():
            raise RuntimeError("Server is not running")

        # Ensure output directory exists.
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert GLTF to GLB if needed. GLTF files may reference external .bin
        # buffers which cannot be sent via HTTP. GLB is self-contained.
        send_path = mesh_path
        temp_glb_path: Path | None = None
        if mesh_path.suffix.lower() == ".gltf":
            temp_glb_path = Path(tempfile.mktemp(suffix=".glb"))
            convert_gltf_to_glb(mesh_path, temp_glb_path)
            send_path = temp_glb_path
            console_logger.debug(
                f"Converted GLTF to GLB for HTTP transfer: {send_path}"
            )

        try:
            # Build form data dict.
            data = {
                "output_dir": str(output_dir),
                "elevation_degrees": str(elevation_degrees),
                "num_side_views": str(num_side_views),
                "include_vertical_views": str(include_vertical_views).lower(),
                "width": str(width),
                "height": str(height),
                "start_azimuth_degrees": str(start_azimuth_degrees),
                "show_coordinate_frame": str(show_coordinate_frame).lower(),
            }
            if light_energy is not None:
                data["light_energy"] = str(light_energy)
            if taa_samples is not None:
                data["taa_samples"] = str(taa_samples)

            # Use retry wrapper with file upload.
            result = self._make_multipart_request_with_retry(
                endpoint="/render_multiview",
                mesh_path=send_path,
                data=data,
                timeout=timeout,
            )
            return [Path(p) for p in result["image_paths"]]
        finally:
            # Clean up temporary GLB file.
            if temp_glb_path is not None and temp_glb_path.exists():
                temp_glb_path.unlink()

    def canonicalize_mesh(
        self,
        input_path: Path,
        output_path: Path,
        up_axis: str,
        front_axis: str,
        object_type: str = "furniture",
        timeout: float = 60.0,
    ) -> Path:
        """Canonicalize mesh via HTTP request to server.

        If the server crashes during canonicalization, it will be automatically
        restarted and the request retried.

        Args:
            input_path: Path to input GLTF file.
            output_path: Path where canonicalized GLTF will be saved.
            up_axis: Up axis in Blender coordinates (e.g., "+Z", "-Y").
            front_axis: Front axis in Blender coordinates (e.g., "+Y", "+X").
            object_type: Type of object (determines placement strategy).
                One of: "furniture", "manipuland", "wall_mounted", "ceiling_mounted".
            timeout: HTTP request timeout in seconds.

        Returns:
            Path to the canonicalized GLTF file.

        Raises:
            RuntimeError: If server is not running or canonicalization fails after
                retries.
        """
        if not self.is_running():
            raise RuntimeError("Server is not running")

        self._make_request_with_retry(
            endpoint="/canonicalize",
            timeout=timeout,
            json={
                "input_path": str(input_path),
                "output_path": str(output_path),
                "up_axis": up_axis,
                "front_axis": front_axis,
                "object_type": object_type,
            },
        )

        return output_path

    def convert_glb_to_gltf(
        self,
        input_path: Path,
        output_path: Path,
        export_yup: bool = True,
        timeout: float = 60.0,
    ) -> Path:
        """Convert GLB to GLTF via HTTP request to server.

        This method delegates the conversion to BlenderServer, ensuring bpy crashes
        don't kill the scene worker process. If the server crashes, it will be
        automatically restarted and the request retried.

        Args:
            input_path: Path to input GLB or GLTF file.
            output_path: Path where converted GLTF will be saved.
            export_yup: If True, converts to Y-up GLTF standard. Default True.
            timeout: HTTP request timeout in seconds.

        Returns:
            Path to the converted GLTF file.

        Raises:
            RuntimeError: If server is not running or conversion fails after retries.
        """
        if not self.is_running():
            raise RuntimeError("Server is not running")

        result = self._make_request_with_retry(
            endpoint="/convert_glb_to_gltf",
            timeout=timeout,
            json={
                "input_path": str(input_path),
                "output_path": str(output_path),
                "export_yup": export_yup,
            },
            result_key="output_path",
        )
        # result_key returns Path for str values.
        assert isinstance(result, Path)
        return result

    def _make_request_with_retry(
        self,
        endpoint: str,
        timeout: float,
        json: dict | None = None,
        files: dict | None = None,
        data: dict | None = None,
        result_key: str | None = None,
        max_retries: int = 2,
    ) -> Path | dict | list:
        """Make HTTP request with automatic server restart on crash.

        If the BlenderServer subprocess crashes (detected via connection failure and
        process death), this method will restart the server and retry the request.

        Args:
            endpoint: The API endpoint to call (e.g., "/convert_glb_to_gltf").
            timeout: HTTP request timeout in seconds.
            json: JSON body to send with the request. Mutually exclusive with
                files/data.
            files: Files dict for multipart form data (e.g., {"mesh": (name, f, type)}).
            data: Form data dict for multipart form data.
            result_key: If provided, return response[result_key] as Path (if str) or
                list of Paths (if list of str).
            max_retries: Maximum number of restart+retry attempts.

        Returns:
            Path, list of Paths, or response dict depending on result_key.

        Raises:
            RuntimeError: If request fails after all retries.
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                url = f"{self.get_url()}{endpoint}"
                if json is not None:
                    response = requests.post(url, json=json, timeout=timeout)
                else:
                    response = requests.post(
                        url, files=files, data=data, timeout=timeout
                    )

                if response.status_code != 200:
                    error_msg = response.text
                    try:
                        error_json = response.json()
                        if "description" in error_json:
                            error_msg = error_json["description"]
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Request to {endpoint} failed ({response.status_code}): "
                        f"{error_msg}"
                    )

                result = response.json()
                if result_key:
                    value = result[result_key]
                    if isinstance(value, list):
                        return [Path(p) for p in value]
                    return Path(value)
                return result

            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                # Check if server process died.
                if self._server_process and self._server_process.poll() is not None:
                    exit_code = self._server_process.returncode
                    if attempt < max_retries:
                        console_logger.warning(
                            f"BlenderServer crashed (exitcode={exit_code}), "
                            f"restarting (attempt {attempt + 1}/{max_retries})"
                        )
                        self._restart_server()
                        continue
                    else:
                        raise RuntimeError(
                            f"BlenderServer crashed (exitcode={exit_code}) and "
                            f"failed after {max_retries} restart attempts: {e}"
                        )
                else:
                    # Server still alive but request failed - don't retry.
                    raise RuntimeError(f"Request to {endpoint} failed: {e}")

        raise RuntimeError(
            f"Request to {endpoint} failed after {max_retries} retries: {last_error}"
        )

    def _make_multipart_request_with_retry(
        self,
        endpoint: str,
        mesh_path: Path,
        data: dict,
        timeout: float,
        max_retries: int = 2,
    ) -> dict:
        """Make multipart file upload request with automatic server restart on crash.

        This method handles file uploads where the file must be re-opened for each
        retry attempt.

        Args:
            endpoint: The API endpoint to call.
            mesh_path: Path to mesh file to upload.
            data: Form data dict.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum number of restart+retry attempts.

        Returns:
            Response JSON dict.

        Raises:
            RuntimeError: If request fails after all retries.
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                url = f"{self.get_url()}{endpoint}"
                # Re-open file for each attempt (file position resets after read).
                with open(mesh_path, "rb") as f:
                    files = {"mesh": ("mesh.glb", f, "application/octet-stream")}
                    response = requests.post(
                        url, files=files, data=data, timeout=timeout
                    )

                if response.status_code != 200:
                    error_msg = response.text
                    try:
                        error_json = response.json()
                        if "description" in error_json:
                            error_msg = error_json["description"]
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Request to {endpoint} failed ({response.status_code}): "
                        f"{error_msg}"
                    )

                return response.json()

            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                # Check if server process died.
                if self._server_process and self._server_process.poll() is not None:
                    exit_code = self._server_process.returncode
                    if attempt < max_retries:
                        console_logger.warning(
                            f"BlenderServer crashed (exitcode={exit_code}), "
                            f"restarting (attempt {attempt + 1}/{max_retries})"
                        )
                        self._restart_server()
                        continue
                    else:
                        raise RuntimeError(
                            f"BlenderServer crashed (exitcode={exit_code}) and "
                            f"failed after {max_retries} restart attempts: {e}"
                        )
                else:
                    # Server still alive but request failed - don't retry.
                    raise RuntimeError(f"Request to {endpoint} failed: {e}")

        raise RuntimeError(
            f"Request to {endpoint} failed after {max_retries} retries: {last_error}"
        )

    def _restart_server(self) -> None:
        """Restart the BlenderServer after a crash.

        Stops the current (crashed) server process and starts a fresh one.
        """
        console_logger.info("Restarting BlenderServer...")
        self.stop()
        self.start()
        self.wait_until_ready(timeout=30.0)
        console_logger.info("BlenderServer restarted successfully")
