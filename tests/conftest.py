"""Pytest configuration and fixtures for scenesmith tests.

This module provides pytest hooks and fixtures that apply across all tests.
"""

import gc
import logging
import sys

from unittest.mock import MagicMock

import pytest

# isort: off
# Import bpy first to avoid OpenGL context conflicts with Drake rendering when
# Blender's Python module is installed. Drake-only tests can run without bpy.
try:
    import bpy  # noqa: F401
except ImportError:
    sys.modules["bpy"] = MagicMock()

try:
    import bmesh  # noqa: F401
except ImportError:
    sys.modules["bmesh"] = MagicMock()

# isort: on

console_logger = logging.getLogger(__name__)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Force garbage collection after each test to clean up Drake C++ objects."""
    gc.collect()
    console_logger.debug(f"Garbage collection completed after test: {item.nodeid}")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Force final garbage collection after all tests complete.

    This ensures Drake C++ objects are cleaned up after all tests complete but
    before pytest exits, preventing hangs during Drake's leak detector cleanup.
    """
    del session, exitstatus  # Unused but required by hookspec.
    gc.collect()
    console_logger.debug("Final garbage collection completed after test session")
