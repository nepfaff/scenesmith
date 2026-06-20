"""Tests for floor-plan material server configuration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

from scenesmith.experiments.base_experiment import BaseExperiment
from scenesmith.floor_plan_agents.tools.materials_resolver import (
    MaterialsConfig,
    MaterialsResolver,
)


class _FakeFloorPlanAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_floor_plan_agent_passes_materials_server_config():
    cfg = OmegaConf.create(
        {
            "floor_plan_agent": {
                "_name": "fake",
                "mode": "room",
            },
            "experiment": {
                "materials_retrieval_server": {
                    "host": "127.0.0.2",
                    "port": 7418,
                },
            },
        }
    )

    agent = BaseExperiment.build_floor_plan_agent(
        cfg,
        compatible_agents={"fake": _FakeFloorPlanAgent},
        logger=SimpleNamespace(),
        render_gpu_id=3,
    )

    assert agent.kwargs["materials_server_host"] == "127.0.0.2"
    assert agent.kwargs["materials_server_port"] == 7418
    assert agent.kwargs["render_gpu_id"] == 3


def test_materials_resolver_uses_configured_retrieval_server(tmp_path: Path):
    client_instances = []

    class FakeClient:
        def __init__(self, host: str, port: int):
            self.host = host
            self.port = port
            client_instances.append(self)

        def retrieve_materials(self, requests):
            assert requests[0].output_dir == str(tmp_path / "materials")
            result = SimpleNamespace(
                material_path=str(tmp_path / "materials" / "oak"),
                material_id="oak",
            )
            yield 0, SimpleNamespace(results=[result])

    config = MaterialsConfig(
        use_retrieval_server=True,
        output_dir=tmp_path,
        retrieval_server_host="127.0.0.2",
        retrieval_server_port=7418,
    )
    resolver = MaterialsResolver(config)

    with patch(
        "scenesmith.floor_plan_agents.tools.materials_resolver."
        "MaterialsRetrievalClient",
        FakeClient,
    ):
        material = resolver.get_material("warm oak flooring")

    assert len(client_instances) == 1
    assert client_instances[0].host == "127.0.0.2"
    assert client_instances[0].port == 7418
    assert material is not None
    assert material.material_id == "oak"
