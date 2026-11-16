from pathlib import Path

from zerre.data import resolve_acoustic_root


def test_override_wins(tmp_path):
    features = tmp_path / "elsewhere"
    features.mkdir()
    manifest = tmp_path / "annotations" / "train.jsonl"
    resolved = resolve_acoustic_root(manifest, {"acoustic_root": "/nope/features"}, features)
    assert resolved == features


def test_environment_wins_over_recorded_path(tmp_path, monkeypatch):
    features = tmp_path / "features_full"
    features.mkdir()
    monkeypatch.setenv("ZERRE_ACOUSTIC_ROOT", str(features))
    resolved = resolve_acoustic_root(tmp_path / "annotations" / "train.jsonl",
                                     {"acoustic_root": "C:\\Users\\someone\\features_full"})
    assert resolved == features


def test_falls_back_to_sibling_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("ZERRE_ACOUSTIC_ROOT", raising=False)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    features = tmp_path / "features_full"
    features.mkdir()
    resolved = resolve_acoustic_root(annotations / "train.jsonl",
                                     {"acoustic_root": "C:\\Users\\someone\\data\\features_full"})
    assert resolved == features


def test_keeps_recorded_path_when_it_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("ZERRE_ACOUSTIC_ROOT", raising=False)
    features = tmp_path / "data" / "features_full"
    features.mkdir(parents=True)
    resolved = resolve_acoustic_root(tmp_path / "annotations" / "train.jsonl",
                                     {"acoustic_root": str(features)})
    assert resolved == Path(str(features))
