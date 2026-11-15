import json
import re
from pathlib import Path

import pytest

from zerre import lang
from zerre.config import Config

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "zerre"


def test_config_derives_vocab_from_language():
    config = Config()
    assert config.language == lang.current_code()
    assert config.vocab_size == lang.current().vocab_size


def test_config_rejects_wrong_vocab_size():
    with pytest.raises(ValueError):
        Config(vocab_size=len(lang.current().symbols) + 1)


def test_config_roundtrip_keeps_speakers_tuple():
    config = Config(speakers=("A", "B"))
    restored = Config.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert isinstance(restored.speakers, tuple)


def test_config_rejects_duplicate_speakers():
    with pytest.raises(ValueError):
        Config(speakers=("A", "A"))


def test_source_tree_has_no_user_specific_paths():
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in (r"C:\\\\Users", r"/Users/", r"/home/"):
            if re.search(pattern, text):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"user-specific paths leaked into the package: {sorted(set(offenders))}"


def test_no_fcntl_import():
    for path in SRC.rglob("*.py"):
        assert "import fcntl" not in path.read_text(encoding="utf-8"), path
