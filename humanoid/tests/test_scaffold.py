"""Mock-only PGAG humanoid scaffold tests; no ROS2/MuJoCo required."""

from humanoid.pgag_unitree_sim import execute_pgag
from humanoid.unitree_g1_adapter import UnitreeG1Adapter


def test_mock_pick_place():
    result = execute_pgag(task="pick_place")
    assert result["success"] is True
    assert len(result["executed_atoms"]) == 5


def test_mock_stack():
    result = execute_pgag(task="stack")
    assert result["success"] is True
    assert len(result["executed_atoms"]) == 12
    assert result["local_repairs"] == 0


def test_mock_displacement_is_recovered():
    result = execute_pgag(task="stack", disturb=True)
    assert result["success"] is True
    assert result["stale_detections"] == 1
    assert result["local_repairs"] == 1
    assert len(result["executed_atoms"]) == 18


def test_mock_displacement_without_repair_fails():
    result = execute_pgag(task="stack", disturb=True, repair=False)
    assert result["success"] is False
    assert result["stale_detections"] == 1


def test_real_adapter_is_not_fake():
    adapter = UnitreeG1Adapter()
    try:
        adapter.reset("stack")
    except NotImplementedError:
        return
    raise AssertionError("An unimplemented G1 adapter must not report success.")
