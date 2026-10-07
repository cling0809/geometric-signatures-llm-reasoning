import os
from pathlib import Path

from geoprobe.utils.paths import cache_dir, exp_dir, project_root, runs_root


def test_defaults_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOPROBE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GEOPROBE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("GEOPROBE_RUNS", str(tmp_path / "runs"))

    assert project_root() == tmp_path / "home"
    assert cache_dir() == tmp_path / "cache"
    assert runs_root() == tmp_path / "runs"


def test_exp_dir_creates_subdirs(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOPROBE_RUNS", str(tmp_path))
    d = exp_dir("2026-05-16_pilot", create=True)
    assert d == tmp_path / "2026-05-16_pilot"
    for sub in ("logs", "trajectories", "metrics", "plots"):
        assert (d / sub).is_dir()


def test_exp_dir_no_create(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOPROBE_RUNS", str(tmp_path))
    d = exp_dir("nope", create=False)
    assert d == tmp_path / "nope"
    assert not d.exists()
