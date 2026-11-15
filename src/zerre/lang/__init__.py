"""Language packs. One language per process; see ``zerre.lang.base``."""

import os

from . import tr
from .base import Language

__all__ = ["LANGUAGES", "Language", "current", "current_code", "get", "set_language"]

LANGUAGES = {"tr": tr.LANGUAGE}
DEFAULT_LANGUAGE = "tr"
LANGUAGE_VARIABLE = "ZERRE_LANG"


def get(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"unknown language {code!r}; available: {sorted(LANGUAGES)}") from None


def current_code() -> str:
    code = os.environ.get(LANGUAGE_VARIABLE, DEFAULT_LANGUAGE)
    if code not in LANGUAGES:
        raise ValueError(f"{LANGUAGE_VARIABLE} must be one of {sorted(LANGUAGES)}, not {code!r}")
    return code


def current() -> Language:
    return LANGUAGES[current_code()]


def set_language(code: str) -> Language:
    """Select the active language and keep ``zerre.text`` in sync.

    Call this before any tensor work. Switching language invalidates token ids
    from another pack, so it is only meant for one-shot CLI runs.
    """
    language = get(code)
    os.environ[LANGUAGE_VARIABLE] = code
    from .. import text

    text._bind(language)
    return language
