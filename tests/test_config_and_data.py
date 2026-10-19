import pytest

from mv3dp.config import PipelineConfig, load_config
from mv3dp.reconstruction.sparse import split_holdout


def test_load_config_with_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("scene_dir: data/x\ndense:\n  method: colmap\n")
    cfg = load_config(path, {"detection.confidence": 0.5, "fusion.min_views": 4})
    assert isinstance(cfg, PipelineConfig)
    assert cfg.scene_dir == "data/x"
    assert cfg.dense.method == "colmap"
    assert cfg.detection.confidence == 0.5
    assert cfg.fusion.min_views == 4
    assert cfg.sparse.holdout_every == 8  # default kept


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("dense:\n  methd: sgbm\n")
    with pytest.raises(KeyError):
        load_config(path)


def test_shipped_configs_load():
    from pathlib import Path

    for path in Path(__file__).parents[1].glob("configs/*.yaml"):
        load_config(path)


def test_holdout_split_matches_mipnerf_convention():
    names = [f"{i:03d}.jpg" for i in range(20)]
    train, test = split_holdout(names, 8)
    assert test == ["000.jpg", "008.jpg", "016.jpg"]
    assert len(train) == 17 and not set(train) & set(test)
    assert split_holdout(names, 0) == (names, [])
