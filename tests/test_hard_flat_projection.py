from __future__ import annotations

import numpy as np

from dataset_new.build_hard_flat_projection_dataset import (
    interval_overlap,
    rectangular_overlap_gate_response,
)


def test_interval_overlap_returns_common_length() -> None:
    assert interval_overlap(0.0, 1.0, 0.0, 1.0) == 1.0
    assert interval_overlap(0.0, 1.0, 0.5, 1.0) == 0.5
    assert interval_overlap(0.0, 1.0, 2.0, 1.0) == 0.0


def test_equal_rectangles_give_triangular_response() -> None:
    centered = rectangular_overlap_gate_response(
        num_gates=3,
        center=1.0,
        pulse_width=1.0,
        gate_width=1.0,
        gate_spacing=1.0,
        min_response=0.0,
    )
    halfway = rectangular_overlap_gate_response(
        num_gates=3,
        center=0.5,
        pulse_width=1.0,
        gate_width=1.0,
        gate_spacing=1.0,
        min_response=0.0,
    )

    np.testing.assert_allclose(centered, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(halfway, np.asarray([0.5, 0.5, 0.0], dtype=np.float32))


def test_unequal_rectangles_can_create_trapezoid_plateau() -> None:
    response = rectangular_overlap_gate_response(
        num_gates=5,
        center=2.0,
        pulse_width=3.0,
        gate_width=1.0,
        gate_spacing=0.5,
        min_response=0.0,
    )

    assert response[2] == 1.0
    assert response[3] == 1.0
    assert response[4] == 1.0
