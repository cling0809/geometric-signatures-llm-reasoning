import math

import torch

from geoprobe.metrics import mean_curvature, per_step_curvature, trajectory_length


def _straight_line(T=10, L=3, H=4):
    """A constant-velocity line in R^H, replicated for L layers. Length = ||v|| * (T-1)."""
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    pts = torch.stack([v * t for t in range(T)])  # [T, H]
    return pts.unsqueeze(1).expand(T, L, H).contiguous()


def _right_angle(L=2, H=4):
    """Three-step zigzag in R^4: (0,0)->(1,0)->(2,0)->(2,1). Angle at step 2 = 90deg."""
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
        ]
    )
    return pts.unsqueeze(1).expand(4, L, H).contiguous()


def test_length_straight_line():
    hs = _straight_line(T=10, L=3, H=4)
    L = trajectory_length(hs)
    assert L.shape == (3,)
    # 9 segments of length 1
    assert torch.allclose(L, torch.full((3,), 9.0), atol=1e-5)


def test_curvature_straight_line_is_zero():
    hs = _straight_line(T=10, L=3, H=4)
    psc = per_step_curvature(hs)
    assert psc.shape == (8, 3)
    assert torch.allclose(psc, torch.zeros_like(psc), atol=1e-3)


def test_curvature_right_angle():
    hs = _right_angle(L=2, H=4)
    psc = per_step_curvature(hs)
    # angles per interior step: step 1 (straight) = 0, step 2 (90 deg) = pi/2
    assert psc.shape == (2, 2)
    assert math.isclose(psc[0, 0].item(), 0.0, abs_tol=1e-3)
    assert math.isclose(psc[1, 0].item(), math.pi / 2, abs_tol=1e-3)


def test_mean_curvature_layer_independent_when_layers_identical():
    hs = _right_angle(L=4, H=4)
    mc = mean_curvature(hs)
    assert mc.shape == (4,)
    # all layers identical -> all means equal
    assert torch.allclose(mc, mc[0].expand_as(mc), atol=1e-6)


def test_length_handles_short_traj():
    hs = torch.zeros((1, 3, 4))
    L = trajectory_length(hs)
    assert L.shape == (3,)
    assert torch.allclose(L, torch.zeros(3))


def test_curvature_handles_short_traj():
    hs = torch.zeros((2, 3, 4))
    psc = per_step_curvature(hs)
    assert psc.shape == (0, 3)


def test_mean_step_norm_is_length_over_T_minus_1():
    from geoprobe.metrics import mean_step_norm

    hs = _straight_line(T=10, L=3, H=4)  # 9 segments of length 1
    msn = mean_step_norm(hs)
    assert msn.shape == (3,)
    assert torch.allclose(msn, torch.ones(3), atol=1e-5)


def test_curvature_stats_dict_shapes():
    from geoprobe.metrics import curvature_stats

    hs = _right_angle(L=2, H=4)
    stats = curvature_stats(hs)
    assert set(stats.keys()) == {"mean", "var", "max", "p90", "p99"}
    for k, v in stats.items():
        assert v.shape == (2,), f"{k} has shape {v.shape}"
    # max should be ~ pi/2 for the right-angle trajectory
    assert torch.allclose(stats["max"], torch.full((2,), math.pi / 2), atol=1e-3)


def test_trajectory_metrics_returns_all_keys():
    from geoprobe.metrics import trajectory_metrics

    hs = _straight_line(T=5, L=2, H=4)
    m = trajectory_metrics(hs)
    expected = {
        "trajectory_length",
        "mean_step_norm",
        "curvature_mean",
        "curvature_var",
        "curvature_max",
        "curvature_p90",
        "curvature_p99",
    }
    assert expected <= set(m.keys())
    for k, v in m.items():
        assert v.shape == (2,), f"{k} has shape {v.shape}"
