from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLENDER_SCRIPT = PROJECT_ROOT / "origindataset" / "gated_blender_physical.py"
RENDER_SCRIPT = PROJECT_ROOT / "scripts" / "render_selected_military_gates.ps1"


def test_blender_script_compiles_without_importing_bpy() -> None:
    source = BLENDER_SCRIPT.read_text(encoding="utf-8")

    compile(source, str(BLENDER_SCRIPT), "exec")


def test_flat_echo_exposes_camera_depth_flattening_mode() -> None:
    source = BLENDER_SCRIPT.read_text(encoding="utf-8")

    assert "--flat-geometry-mode" in source
    assert "--flat-target-gate-index-mode" in source
    assert "round-robin" in source
    assert "flatten-camera-depth" in source
    assert "resolve_flat_target_gate_index" in source
    assert "flatten_object_to_camera_depth(obj, flat_depth)" in source
    assert "flat_target_gate_index_mode" in source
    assert "flat_geometry_depth_min" in source
    assert "flat_geometry_depth_max" in source
    assert "--reflectance-mode" in source
    assert "hash-log-uniform" in source
    assert "target_reflectance" in source
    assert "TargetReflectance" in source
    assert "reflectance_for_sample" in source


def test_cli_relative_paths_are_resolved_against_project_root() -> None:
    source = BLENDER_SCRIPT.read_text(encoding="utf-8")

    assert "def resolve_project_path(path):" in source
    assert "return SCRIPT_ROOT / path" in source
    assert "output_root = resolve_project_path(args.output_root)" in source
    assert "resolve_project_path(args.single_model)" in source


def test_selected_military_renderer_uses_flattening_by_default() -> None:
    source = RENDER_SCRIPT.read_text(encoding="utf-8")

    assert '[string]$FlatGeometryMode = "flatten-camera-depth"' in source
    assert "[double]$ReceiverGateWidth = 0.9" in source
    assert "[double]$LaserPulseWidth = 0.45" in source
    assert "[double]$AutoGateMargin = 0.08" in source
    assert '"--receiver-gate-width", ([string]$ReceiverGateWidth)' in source
    assert '"--laser-pulse-width", ([string]$LaserPulseWidth)' in source
    assert '"--auto-gate-margin", ([string]$AutoGateMargin)' in source
    assert '"--flat-geometry-mode", $FlatGeometryMode' in source
    assert '"--reflectance-mode", $ReflectanceMode' in source
    assert '[string]$ModelRotationDeg = "0,0,0"' in source
    assert '"--model-rotation-deg", $ModelRotationDeg' in source
