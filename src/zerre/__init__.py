"""zerre-tts: a small, language-packed TTS trained from scratch.

The speech model loads no pretrained weights. A pretrained phoneme recogniser
is used only by the annotation step, and an ASR model only by evaluation.
"""

from .config import Config
from .model import TinyVoca

__all__ = ["Config", "TinyVoca"]
__version__ = "0.1.0"
