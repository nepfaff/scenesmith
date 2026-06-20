import hashlib
import io
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import flask

from mathutils import Vector

from scenesmith.agent_utils.blender import (
    BlenderRenderApp,
    BlenderRenderer,
    BlenderServer,
    RenderParams,
)
from scenesmith.agent_utils.blender.annotations import annotate_image_with_coordinates
from scenesmith.agent_utils.blender.coordinate_frame import create_coordinate_frame
from scenesmith.agent_utils.blender.scene_utils import get_floor_bounds
from scenesmith.agent_utils.blender.server_manager import (
    find_available_port,
    is_port_available,
)


class TestRenderParams(unittest.TestCase):
    """Test cases for RenderParams dataclass."""

    def test_render_params_creation(self):
        """Test creating RenderParams with required fields."""
        params = RenderParams(
            scene=Path("/tmp/test.gltf"),
            scene_sha256="abc123",
            image_type="color",
            width=640,
            height=480,
            near=0.1,
            far=100.0,
            focal_x=320.0,
            focal_y=320.0,
            fov_x=1.047,
            fov_y=0.785,
            center_x=320.0,
            center_y=240.0,
        )

        self.assertEqual(params.scene, Path("/tmp/test.gltf"))
        self.assertEqual(params.scene_sha256, "abc123")
        self.assertEqual(params.image_type, "color")
        self.assertEqual(params.width, 640)
        self.assertEqual(params.height, 480)
        self.assertEqual(params.near, 0.1)
        self.assertEqual(params.far, 100.0)
        self.assertEqual(params.focal_x, 320.0)
        self.assertEqual(params.focal_y, 320.0)
        self.assertEqual(params.fov_x, 1.047)
        self.assertEqual(params.fov_y, 0.785)
        self.assertEqual(params.center_x, 320.0)
        self.assertEqual(params.center_y, 240.0)
        self.assertIsNone(params.min_depth)
        self.assertIsNone(params.max_depth)


class TestBlenderRenderer(unittest.TestCase):
    """Test cases for BlenderRenderer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.blend_file = self.temp_dir / "test.blend"
        self.settings_file = self.temp_dir / "settings.py"

    def test_blender_renderer_init(self):
        """Test BlenderRenderer initialization."""
        renderer = BlenderRenderer()
        self.assertIsNone(renderer._blend_file)
        self.assertIsNone(renderer._bpy_settings_file)
        self.assertIsNone(renderer._client_objects)

    def test_blender_renderer_init_with_files(self):
        """Test BlenderRenderer initialization with blend and settings files."""
        renderer = BlenderRenderer(
            blend_file=self.blend_file,
            bpy_settings_file=self.settings_file,
        )
        self.assertEqual(renderer._blend_file, self.blend_file)
        self.assertEqual(renderer._bpy_settings_file, self.settings_file)

    @patch("scenesmith.agent_utils.blender.scene_setup_mixin.bpy")
    @patch("scenesmith.agent_utils.blender.renderer.bpy")
    def test_reset_scene(self, mock_renderer_bpy, mock_setup_bpy):
        """Test that reset_scene resets Blender scene and removes default objects."""
        # Mock the data objects in scene_setup_mixin (where reset_scene is defined).
        mock_setup_bpy.data.objects = [Mock(), Mock()]

        renderer = BlenderRenderer()
        renderer.reset_scene()

        # Should call factory settings read and delete objects.
        mock_setup_bpy.ops.wm.read_factory_settings.assert_called_once()
        mock_setup_bpy.ops.object.delete.assert_called_once()

        # Should select each object.
        for obj in mock_setup_bpy.data.objects:
            obj.select_set.assert_called_with(True)

    @patch("scenesmith.agent_utils.blender.renderer.bpy")
    def test_add_default_light_source(self, mock_bpy):
        """Test that add_default_light_source adds a point light."""
        mock_light = MagicMock()
        mock_light_object = MagicMock()
        mock_bpy.data.lights.new.return_value = mock_light
        mock_bpy.data.objects.new.return_value = mock_light_object

        renderer = BlenderRenderer()
        renderer.add_default_light_source()

        # Should create light and light object.
        mock_bpy.data.lights.new.assert_called_once_with(
            name="DefaultLight", type="POINT"
        )
        mock_bpy.data.objects.new.assert_called_once_with(
            name="DefaultLight", object_data=mock_light
        )
        self.assertEqual(mock_light.energy, 1000)
        self.assertEqual(mock_light_object.location, (4.0, 1.0, 6.0))

    @patch("scenesmith.agent_utils.blender.scene_setup_mixin.bpy")
    @patch("scenesmith.agent_utils.blender.render_settings.bpy")
    @patch("scenesmith.agent_utils.blender.camera_utils.bpy")
    @patch("scenesmith.agent_utils.blender.renderer.bpy")
    def test_render_image_creates_output_file(
        self, mock_renderer_bpy, mock_camera_bpy, mock_settings_bpy, mock_setup_bpy
    ):
        """Test that render_image creates rendered output file."""
        # Create mock scene file with correct checksum.
        test_scene_content = b"test gltf content"
        test_scene_path = Path("/tmp/test.gltf")

        # Mock the file reading and bpy imports.
        with patch.object(Path, "read_bytes", return_value=test_scene_content):
            # Calculate correct checksum.
            correct_sha256 = hashlib.sha256(test_scene_content).hexdigest()

            renderer = BlenderRenderer()
            params = RenderParams(
                scene=test_scene_path,
                scene_sha256=correct_sha256,
                image_type="color",
                width=640,
                height=480,
                near=0.1,
                far=100.0,
                focal_x=320.0,
                focal_y=320.0,
                fov_x=1.047,
                fov_y=0.785,
                center_x=320.0,
                center_y=240.0,
            )
            output_path = Path("/tmp/output.png")

            # Mock scene and camera objects for all modules.
            mock_scene = Mock()
            mock_scene.render = Mock()
            mock_scene.render.resolution_x = None
            mock_scene.render.resolution_y = None
            mock_scene.render.filepath = None

            # Mock world nodes for setup_regular_world().
            mock_world = Mock()
            mock_world.use_nodes = False
            mock_scene.world = mock_world
            mock_bg_node = Mock()
            mock_bg_input = Mock()
            mock_bg_input.default_value = None
            mock_bg_node.inputs = {0: mock_bg_input}
            mock_node_tree = Mock()
            mock_node_tree.nodes.get = Mock(return_value=mock_bg_node)
            mock_world.node_tree = mock_node_tree

            mock_camera_data = Mock()
            mock_camera_object = Mock()

            # Mock collections for GLTF import (_import_and_organize_gltf).
            mock_collection = Mock()
            mock_collection.objects = (
                []
            )  # Must be iterable for disable_backface_culling.
            mock_setup_bpy.data.collections.new.return_value = mock_collection
            mock_setup_bpy.context.selected_objects = []

            # Set up bpy mocks for all modules.
            for mock_bpy in [
                mock_renderer_bpy,
                mock_camera_bpy,
                mock_settings_bpy,
                mock_setup_bpy,
            ]:
                mock_bpy.context.scene = mock_scene
                mock_bpy.data.cameras.new.return_value = mock_camera_data
                mock_bpy.data.objects.new.return_value = mock_camera_object
                mock_bpy.data.worlds.new = Mock(return_value=mock_world)

            renderer.render_image(params, output_path)

            # Should call import of gltf scene in scene_setup_mixin.
            mock_setup_bpy.ops.import_scene.gltf.assert_called_once_with(
                filepath=str(test_scene_path)
            )

            # Should set render parameters.
            self.assertEqual(mock_scene.render.resolution_x, 640)
            self.assertEqual(mock_scene.render.resolution_y, 480)
            self.assertEqual(mock_scene.render.filepath, str(output_path))

            # Should call render.
            mock_renderer_bpy.ops.render.render.assert_called_once_with(
                write_still=True
            )

            # Should set up collections and rotation in scene_setup_mixin.
            mock_setup_bpy.data.collections.new.assert_called()
            mock_setup_bpy.ops.transform.rotate.assert_called_once()


class TestBlenderRenderApp(unittest.TestCase):
    """Test cases for BlenderRenderApp Flask application."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_blender_render_app_init(self):
        """Test BlenderRenderApp initialization."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)
        self.assertEqual(app.name, "scenesmith_blender_render")
        self.assertEqual(app._temp_dir, self.temp_dir)
        self.assertIsInstance(app._blender, BlenderRenderer)

    def test_blender_render_app_init_with_files(self):
        """Test BlenderRenderApp initialization with blend and settings files."""
        blend_file = Path("/tmp/test.blend")
        settings_file = Path("/tmp/settings.py")

        app = BlenderRenderApp(
            temp_dir=self.temp_dir,
            blend_file=blend_file,
            bpy_settings_file=settings_file,
        )
        self.assertEqual(app._blender._blend_file, blend_file)
        self.assertEqual(app._blender._bpy_settings_file, settings_file)

    def test_root_endpoint_returns_banner(self):
        """Test that _root_endpoint returns HTML banner page."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)
        response = app._root_endpoint()

        self.assertIsInstance(response, str)
        self.assertIn("<!doctype html>", response.lower())
        self.assertIn("blender", response.lower())

    def test_render_endpoint_handles_post_request(self):
        """Test that _render_endpoint handles POST requests and returns image."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)

        with app.test_request_context("/render", method="POST"):
            with patch.object(app, "_parse_params") as mock_parse:
                with patch.object(app, "_render") as mock_render:
                    mock_buffer = io.BytesIO(b"fake_png_data")
                    mock_render.return_value = mock_buffer

                    response = app._render_endpoint()

                    mock_parse.assert_called_once()
                    mock_render.assert_called_once()
                    self.assertIsInstance(response, flask.Response)

    def test_parse_params_converts_form_data(self):
        """Test that _parse_params correctly parses Flask request form data."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)

        # Mock request with form data.
        mock_request = Mock(spec=flask.Request)
        mock_request.form = {
            "scene_sha256": "abc123",
            "image_type": "color",
            "width": "640",
            "height": "480",
            "near": "0.1",
            "far": "100.0",
            "focal_x": "320.0",
            "focal_y": "320.0",
            "fov_x": "1.047",
            "fov_y": "0.785",
            "center_x": "320.0",
            "center_y": "240.0",
        }
        mock_files = {"scene": Mock()}

        # Mock save method to create actual file for stat() call.
        def mock_save(path):
            Path(path).touch()

        mock_files["scene"].save = mock_save
        mock_request.files = mock_files

        params = app._parse_params(mock_request)

        self.assertIsInstance(params, RenderParams)
        self.assertEqual(params.scene_sha256, "abc123")
        self.assertEqual(params.image_type, "color")
        self.assertEqual(params.width, 640)
        self.assertEqual(params.height, 480)

    def test_render_returns_png_buffer(self):
        """Test that _render calls BlenderRenderer and returns PNG buffer."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)
        params = RenderParams(
            scene=Path("/tmp/test.gltf"),
            scene_sha256="abc123",
            image_type="color",
            width=640,
            height=480,
            near=0.1,
            far=100.0,
            focal_x=320.0,
            focal_y=320.0,
            fov_x=1.047,
            fov_y=0.785,
            center_x=320.0,
            center_y=240.0,
        )

        with patch.object(app._blender, "render_image") as mock_render:
            # Mock the temporary file creation and PNG data.
            mock_png_data = b"fake_png_data"

            with (
                patch("tempfile.NamedTemporaryFile") as mock_tempfile,
                patch.object(Path, "read_bytes", return_value=mock_png_data),
                patch.object(Path, "exists", return_value=True),
                patch.object(Path, "unlink"),
            ):

                # Setup the mock temp file.
                mock_temp = MagicMock()
                mock_temp.name = "/tmp/tmpfile.png"
                mock_tempfile.return_value.__enter__.return_value = mock_temp

                buffer = app._render(params)

                self.assertIsInstance(buffer, io.BytesIO)
                # Verify render_image was called with params and some temp path.
                self.assertEqual(mock_render.call_count, 1)
                call_args = mock_render.call_args[0]
                self.assertEqual(call_args[0], params)
                self.assertIsInstance(call_args[1], Path)

    def test_url_rules_configured(self):
        """Test that URL rules are properly configured."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)

        # Check that routes are registered.
        rule_endpoints = [rule.endpoint for rule in app.url_map.iter_rules()]
        self.assertIn("/render", rule_endpoints)

        # Check that POST method is allowed for render endpoint.
        render_rule = None
        for rule in app.url_map.iter_rules():
            if rule.endpoint == "/render":
                render_rule = rule
                break

        self.assertIsNotNone(render_rule)
        self.assertIn("POST", render_rule.methods)


class TestBlenderServer(unittest.TestCase):
    """Test cases for BlenderServer lifecycle manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.blend_file = self.temp_dir / "test.blend"
        self.settings_file = self.temp_dir / "settings.py"

    def test_blender_server_init(self):
        """Test BlenderServer initialization with default parameters."""
        server = BlenderServer()
        self.assertEqual(server._host, "127.0.0.1")
        self.assertIsNone(server._port)
        self.assertEqual(server._port_range, (8000, 8050))
        self.assertIsNone(server._actual_port)
        self.assertIsNone(server._blend_file)
        self.assertIsNone(server._bpy_settings_file)
        self.assertEqual(server._server_startup_delay, 3.0)
        self.assertEqual(server._port_cleanup_delay, 2.0)
        self.assertIsNone(server._server_process)
        self.assertIsNone(server._temp_dir)
        self.assertFalse(server._running)

    def test_blender_server_init_custom_params(self):
        """Test BlenderServer initialization with custom parameters."""
        # Create the files so they exist for validation.
        self.blend_file.touch()
        self.settings_file.touch()

        server = BlenderServer(
            host="192.168.1.100",
            port=9000,
            blend_file=self.blend_file,
            bpy_settings_file=self.settings_file,
        )
        self.assertEqual(server._host, "192.168.1.100")
        self.assertEqual(server._port, 9000)
        self.assertIsNone(server._port_range)
        self.assertIsNone(server._actual_port)
        self.assertEqual(server._blend_file, self.blend_file)
        self.assertEqual(server._bpy_settings_file, self.settings_file)

    def test_blender_server_init_invalid_files(self):
        """Test BlenderServer initialization with non-existent files raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BlenderServer(blend_file=Path("/nonexistent/file.blend"))
        self.assertIn("Blend file not found", str(context.exception))

        with self.assertRaises(ValueError) as context:
            BlenderServer(bpy_settings_file=Path("/nonexistent/settings.py"))
        self.assertIn("Bpy settings file not found", str(context.exception))

    def test_blender_server_init_with_port_range(self):
        """Test BlenderServer initialization with port range."""
        server = BlenderServer(port_range=(9000, 9005))
        self.assertEqual(server._host, "127.0.0.1")
        self.assertIsNone(server._port)
        self.assertEqual(server._port_range, (9000, 9005))
        self.assertIsNone(server._actual_port)

    def test_blender_server_init_both_port_and_range_raises_error(self):
        """Test BlenderServer initialization with both port and port_range raises
        ValueError."""
        with self.assertRaises(ValueError) as context:
            BlenderServer(port=8000, port_range=(9000, 9005))
        self.assertIn("Cannot specify both port and port_range", str(context.exception))

    def test_is_running_initial_state(self):
        """Test that server is not running initially."""
        server = BlenderServer()
        self.assertFalse(server.is_running())

    def test_get_url_when_not_running(self):
        """Test that get_url raises RuntimeError when server is not running."""
        server = BlenderServer()
        with self.assertRaises(RuntimeError) as context:
            server.get_url()
        self.assertIn("Server is not running", str(context.exception))
        self.assertIn("status:", str(context.exception))

    def test_get_url_when_running(self):
        """Test get_url returns correct URL when server is marked as running."""
        server = BlenderServer(host="localhost", port=8080)
        server._running = True  # Simulate running state
        server._actual_port = 8080  # Set the actual port for URL generation
        url = server.get_url()
        self.assertEqual(url, "http://localhost:8080")

    @patch(
        "scenesmith.agent_utils.blender.server_manager.is_port_available",
        return_value=True,
    )
    @patch("tempfile.TemporaryDirectory")
    @patch(
        "scenesmith.agent_utils.blender.server_manager.BlenderServer._python_can_import_bpy",
        return_value=True,
    )
    @patch("subprocess.Popen")
    @patch.object(Path, "exists", return_value=True)  # Mock standalone script exists
    def test_start_creates_process(
        self,
        mock_exists,
        mock_popen,
        mock_can_import_bpy,
        mock_temp_dir,
        mock_port_available,
    ):
        """Test that start creates temporary directory and process."""
        mock_temp_dir_instance = Mock()
        mock_temp_dir_instance.name = "/tmp/test"
        mock_temp_dir.return_value = mock_temp_dir_instance

        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process is still running
        mock_process.communicate.return_value = ("", "")  # Mock stdout/stderr
        mock_popen.return_value = mock_process

        server = BlenderServer(
            port=8000, server_startup_delay=0.0, port_cleanup_delay=0.0
        )
        server.start()

        # Should create temporary directory.
        mock_temp_dir.assert_called_once()

        # Should create and start process.
        mock_popen.assert_called_once()

        # Verify command contains expected parts.
        call_args = mock_popen.call_args[0][0]  # First positional arg
        self.assertIn("python", call_args[0])  # sys.executable contains "python"
        self.assertIn("standalone_server.py", call_args[1])
        self.assertIn("--host", call_args)
        self.assertIn("--port", call_args)
        self.assertIn("8000", call_args)  # Check specific port is used

        # Should mark as running.
        self.assertTrue(server.is_running())
        # Should set actual port.
        self.assertEqual(server._actual_port, 8000)

    @patch("tempfile.TemporaryDirectory")
    def test_stop_cleans_up_resources(self, mock_temp_dir):
        """Test that stop cleans up process and temporary directory."""
        mock_temp_dir_instance = Mock()
        mock_temp_dir.return_value = mock_temp_dir_instance

        server = BlenderServer(server_startup_delay=0.0, port_cleanup_delay=0.0)

        # Simulate running state.
        server._running = True
        server._temp_dir = mock_temp_dir_instance
        server._server_process = Mock()
        server._server_process.pid = 12345
        server._server_process.wait.return_value = 0

        # Capture the mock process before calling stop (since stop sets it to None).
        mock_process = server._server_process

        server.stop()

        # Should terminate process and cleanup.
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5.0)
        mock_temp_dir_instance.cleanup.assert_called_once()

        # Should mark as not running and clear references.
        self.assertFalse(server.is_running())
        self.assertIsNone(server._server_process)
        self.assertIsNone(server._temp_dir)

    def test_get_process_status_no_process(self):
        """Test get_process_status when no process exists."""
        server = BlenderServer()
        status = server.get_process_status()
        self.assertEqual(status, "No process")

    def test_get_process_status_running(self):
        """Test get_process_status when process is running."""
        server = BlenderServer()
        server._server_process = Mock()
        server._server_process.pid = 12345
        server._server_process.poll.return_value = None  # Still running

        status = server.get_process_status()
        self.assertEqual(status, "Running (PID 12345)")

    def test_get_process_status_exited(self):
        """Test get_process_status when process has exited."""
        server = BlenderServer()
        server._server_process = Mock()
        server._server_process.poll.return_value = 0  # Exited with code 0

        status = server.get_process_status()
        self.assertEqual(status, "Exited with code 0")


class TestPortUtilities(unittest.TestCase):
    """Test cases for port availability utility functions."""

    def test_is_port_available_free_port(self):
        """Test is_port_available returns True for free ports."""

        # Test with a very high port that should be available.
        self.assertTrue(is_port_available("127.0.0.1", 65432))

    def test_find_available_port_finds_port(self):
        """Test find_available_port finds an available port in range."""
        # Use a high port range that should have available ports.
        port = find_available_port("127.0.0.1", (65400, 65410))
        self.assertIsNotNone(port)
        self.assertGreaterEqual(port, 65400)
        self.assertLessEqual(port, 65410)

    def test_find_available_port_returns_none_when_no_ports(self):
        """Test find_available_port returns None when no ports available."""
        # Mock socket.socket to always raise OSError (port unavailable).
        with patch("socket.socket") as mock_socket:
            mock_context = mock_socket.return_value.__enter__
            mock_context.return_value.bind.side_effect = OSError(
                "Address already in use"
            )

            port = find_available_port("127.0.0.1", (8000, 8002))
            self.assertIsNone(port)


class TestMetricRendering(unittest.TestCase):
    """Test cases for metric rendering functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_render_endpoint_uses_standard_rendering(self):
        """Test that /render endpoint uses standard rendering only."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)

        # Mock the parse_params to return standard image type.
        mock_params = RenderParams(
            scene=Path("/tmp/test.gltf"),
            scene_sha256="abc123",
            image_type="color",
            width=640,
            height=480,
            near=0.1,
            far=100.0,
            focal_x=320.0,
            focal_y=320.0,
            fov_x=1.047,
            fov_y=0.785,
            center_x=320.0,
            center_y=240.0,
        )

        with app.test_request_context("/render", method="POST"):
            with patch.object(app, "_parse_params", return_value=mock_params):
                with patch.object(app, "_render") as mock_standard_render:
                    mock_buffer = io.BytesIO(b"fake_png_data")
                    mock_standard_render.return_value = mock_buffer

                    app._render_endpoint()

                    # Standard endpoint should only call standard rendering.
                    mock_standard_render.assert_called_once_with(mock_params)

    @patch("scenesmith.agent_utils.blender.coordinate_frame.bpy")
    def test_metric_overlays_add_coordinate_frame_and_grid(self, mock_bpy):
        """Test that metric overlays add coordinate frame and grid markers."""
        bbox_center = Vector((0, 0, 0))
        max_dim = 10.0

        # Mock Blender primitive operations and object creation.
        mock_objects = []

        def create_mock_object(name_prefix):
            mock_obj = Mock()
            mock_obj.name = f"{name_prefix}_{len(mock_objects)}"
            mock_obj.data = Mock()
            mock_obj.data.materials = Mock()
            mock_obj.data.materials.append = Mock()
            mock_obj.rotation_mode = "QUATERNION"
            mock_obj.rotation_quaternion = Mock()
            mock_objects.append(mock_obj)
            return mock_obj

        # Mock primitive creation operations.
        def mock_cylinder_add(**_kwargs):
            obj = create_mock_object("cylinder")
            mock_bpy.context.active_object = obj

        def mock_cone_add(**_kwargs):
            obj = create_mock_object("cone")
            mock_bpy.context.active_object = obj

        def mock_text_add(**_kwargs):
            obj = create_mock_object("text")
            mock_bpy.context.active_object = obj

        mock_bpy.ops.mesh.primitive_cylinder_add = mock_cylinder_add
        mock_bpy.ops.mesh.primitive_cone_add = mock_cone_add
        mock_bpy.ops.object.text_add = mock_text_add

        # Mock camera and scene.
        mock_camera = Mock()
        mock_camera.location = Vector((0, 0, 10))
        mock_scene = Mock()
        mock_scene.camera = mock_camera
        mock_bpy.context.scene = mock_scene

        # Mock materials with proper node structure.
        mock_material = Mock()
        mock_material.use_nodes = True
        mock_material.node_tree = Mock()
        mock_bsdf = Mock()
        mock_bsdf.inputs = Mock()
        mock_bsdf.inputs.__getitem__ = Mock(return_value=Mock())
        mock_material.node_tree.nodes = Mock()
        mock_material.node_tree.nodes.get = Mock(return_value=mock_bsdf)
        mock_bpy.data.materials.new = Mock(return_value=mock_material)
        mock_bpy.context.collection.objects.link = Mock()

        # Call the extracted function directly.
        create_coordinate_frame(
            position=bbox_center,
            max_dim=max_dim,
        )

        # Should create coordinate frame objects (3 axes, each with shaft + tip = 6
        # objects minimum).
        self.assertGreaterEqual(len(mock_objects), 6)

        # Objects are automatically added to the active collection by bpy.ops commands.
        # Verify that materials were created for the coordinate frames (one per axis).
        self.assertGreaterEqual(mock_bpy.data.materials.new.call_count, 3)

    @patch("scenesmith.agent_utils.blender.camera_utils.world_to_camera_view")
    def test_strategic_marker_placement_generates_exactly_nine_markers(
        self, mock_world_to_camera
    ):
        """Test that strategic marker placement generates exactly 9 markers."""
        renderer = BlenderRenderer()

        # Ensure _surface_corners is None to use floor bounds mode.
        renderer._surface_corners = None

        # Mock client objects for floor bounds.
        mock_mesh = Mock()
        mock_mesh.type = "MESH"
        mock_mesh.bound_box = [
            (-3, -2, 0),
            (3, -2, 0),
            (-3, 2, 0),
            (3, 2, 0),
            (-3, -2, 2),
            (3, -2, 2),
            (-3, 2, 2),
            (3, 2, 2),
        ]
        mock_mesh.matrix_world = Mock()
        mock_mesh.matrix_world.__matmul__ = lambda self, corner: Vector(corner)

        mock_objects = Mock()
        mock_objects.objects = [mock_mesh]
        renderer._client_objects = mock_objects

        # Mock world_to_camera_view to return normalized coords.
        mock_world_to_camera.return_value = Vector((0.5, 0.5, 1))

        # Mock scene with render resolution.
        mock_scene = Mock()
        mock_scene.render.resolution_x = 1920
        mock_scene.render.resolution_y = 1080
        mock_camera = Mock()
        visual_marks = renderer._get_visual_marks(mock_scene, mock_camera)

        # Should generate exactly 9 strategic positions.
        self.assertEqual(len(visual_marks), 9)

        # Check that we have the expected coordinate positions.
        expected_positions = [
            (-3.0, -2.0),  # bottom-left
            (3.0, -2.0),  # bottom-right
            (-3.0, 2.0),  # top-left
            (3.0, 2.0),  # top-right
            (0.0, -2.0),  # bottom-center
            (0.0, 2.0),  # top-center
            (-3.0, 0.0),  # left-center
            (3.0, 0.0),  # right-center
            (0.0, 0.0),  # center
        ]

        actual_positions = set(visual_marks.keys())
        expected_positions_set = set(expected_positions)
        self.assertEqual(actual_positions, expected_positions_set)

    @patch("scenesmith.agent_utils.blender.camera_utils.world_to_camera_view")
    def test_half_meter_precision_rounding(self, mock_world_to_camera):
        """Test that strategic markers use floor bounds directly without rounding."""
        renderer = BlenderRenderer()

        # Ensure _surface_corners is None to use floor bounds mode.
        renderer._surface_corners = None

        # Mock client objects with floor bounds.
        mock_mesh = Mock()
        mock_mesh.type = "MESH"
        mock_mesh.bound_box = [
            (-3.7, -2.3, 0),
            (3.7, -2.3, 0),
            (-3.7, 2.3, 0),
            (3.7, 2.3, 0),
            (-3.7, -2.3, 2),
            (3.7, -2.3, 2),
            (-3.7, 2.3, 2),
            (3.7, 2.3, 2),
        ]
        mock_mesh.matrix_world = Mock()
        mock_mesh.matrix_world.__matmul__ = lambda self, corner: Vector(corner)

        mock_objects = Mock()
        mock_objects.objects = [mock_mesh]
        renderer._client_objects = mock_objects

        # Mock world_to_camera_view to return normalized coords.
        mock_world_to_camera.return_value = Vector((0.5, 0.5, 1))

        # Mock scene with render resolution.
        mock_scene = Mock()
        mock_scene.render.resolution_x = 1920
        mock_scene.render.resolution_y = 1080
        mock_camera = Mock()
        visual_marks = renderer._get_visual_marks(mock_scene, mock_camera)

        # Should generate 9 markers.
        self.assertEqual(len(visual_marks), 9)

        # Coordinates should be derived from floor bounds (not rounded).
        # With floor bounds [-3.7, -2.3, 0, 3.7, 2.3], we expect:
        # center_x = 0.0, center_y = 0.0
        expected_x_values = {-3.7, 0.0, 3.7}
        expected_y_values = {-2.3, 0.0, 2.3}

        actual_x_values = {x for x, _ in visual_marks.keys()}
        actual_y_values = {y for _, y in visual_marks.keys()}

        # Check that x and y values match expectations (with floating point tolerance).
        for expected_x in expected_x_values:
            self.assertTrue(
                any(
                    abs(actual_x - expected_x) < 0.0001 for actual_x in actual_x_values
                ),
                f"Expected x value {expected_x} not found in {actual_x_values}",
            )

        for expected_y in expected_y_values:
            self.assertTrue(
                any(
                    abs(actual_y - expected_y) < 0.0001 for actual_y in actual_y_values
                ),
                f"Expected y value {expected_y} not found in {actual_y_values}",
            )

    def test_floor_bounds_detection_with_mesh_objects(self):
        """Test floor bounds detection from mesh objects."""
        # Mock mesh objects with bounding boxes.
        mock_mesh1 = Mock()
        mock_mesh1.type = "MESH"
        mock_mesh1.bound_box = [
            (-2, -1, 0),
            (2, -1, 0),
            (-2, 1, 0),
            (2, 1, 0),
            (-2, -1, 2),
            (2, -1, 2),
            (-2, 1, 2),
            (2, 1, 2),
        ]
        mock_mesh1.matrix_world = Mock()
        mock_mesh1.matrix_world.__matmul__ = lambda self, corner: Vector(corner)
        mock_mesh1.users_collection = []

        mock_mesh2 = Mock()
        mock_mesh2.type = "MESH"
        mock_mesh2.bound_box = [
            (-1, -3, 1),
            (1, -3, 1),
            (-1, 3, 1),
            (1, 3, 1),
            (-1, -3, 3),
            (1, -3, 3),
            (-1, 3, 3),
            (1, 3, 3),
        ]
        mock_mesh2.matrix_world = Mock()
        mock_mesh2.matrix_world.__matmul__ = lambda self, corner: Vector(corner)
        mock_mesh2.users_collection = []

        # Mock client objects.
        mock_objects = Mock()
        mock_objects.objects = [mock_mesh1, mock_mesh2]

        floor_bounds = get_floor_bounds(mock_objects)

        # Should find the lowest Z (floor level) and compute 2D bounds.
        self.assertEqual(len(floor_bounds), 5)
        min_x, min_y, floor_z, max_x, max_y = floor_bounds

        # Floor should be at Z=0 (lowest geometry).
        self.assertEqual(floor_z, 0)

        # 2D bounds should encompass all floor-level geometry.
        self.assertEqual(min_x, -2)  # Most negative X from mesh1
        self.assertEqual(max_x, 2)  # Most positive X from mesh1
        self.assertEqual(min_y, -1)  # Most negative Y from mesh1
        self.assertEqual(max_y, 1)  # Most positive Y from mesh1

    def test_floor_bounds_fallback_when_no_objects(self):
        """Test floor bounds raises ValueError when no mesh objects exist."""
        # No client objects should raise ValueError (fail-fast).
        with self.assertRaises(ValueError) as context:
            get_floor_bounds(None)

        self.assertIn("No client objects available", str(context.exception))

    @patch("scenesmith.agent_utils.blender.coordinate_frame.bpy")
    def test_camera_distance_fallback_in_test_environment(self, mock_bpy):
        """Test camera distance calculation fallback for test environments."""
        # Mock camera with invalid location (empty).
        mock_camera = Mock()
        mock_camera.location = []  # Empty location causes ValueError
        mock_scene = Mock()
        mock_scene.camera = mock_camera
        mock_bpy.context.scene = mock_scene

        bbox_center = Vector((0, 0, 0))
        max_dim = 10.0

        # Should not raise exception and use fallback distance.
        try:
            create_coordinate_frame(
                position=bbox_center,
                max_dim=max_dim,
            )
            # If we get here, fallback worked correctly
            self.assertTrue(True)
        except (ValueError, AttributeError):
            self.fail("Should use fallback camera distance in test environment")

    def test_coordinate_formatting_removes_unnecessary_decimals(self):
        """Test that coordinate formatting produces clean text without trailing zeros."""
        # Test the formatting logic used in coordinate display.
        test_cases = [
            (5.0, "5"),  # Should remove .0
            (5.5, "5.5"),  # Should keep .5
            (10.0, "10"),  # Should remove .0
            (3.25, "3.25"),  # Should keep .25
            (-2.0, "-2"),  # Should remove .0 for negatives
            (-1.5, "-1.5"),  # Should keep .5 for negatives
        ]

        for input_val, expected_str in test_cases:
            formatted_str = f"{input_val:g}"
            self.assertEqual(
                formatted_str,
                expected_str,
                f"Expected {input_val} to format as '{expected_str}', got "
                f"'{formatted_str}'",
            )

    def test_coordinate_annotation_method_exists_and_callable(self):
        """Test that coordinate annotation function exists and is callable."""
        # Test that the extracted function exists and can be called.
        self.assertTrue(callable(annotate_image_with_coordinates))

        # Test basic functionality without complex font mocking.
        with patch("PIL.Image.open") as mock_open:
            mock_pil_image = Mock()
            mock_pil_image.mode = "RGB"
            mock_pil_image.size = (100, 100)
            mock_open.return_value = mock_pil_image

            # Should not raise an exception when given empty marks.
            try:
                annotate_image_with_coordinates(
                    image_path=Path("/tmp/test.png"), marks={}
                )
                # If we get here, the function handled empty marks correctly.
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Function should handle empty visual marks gracefully: {e}")


if __name__ == "__main__":
    unittest.main()
