"""
Dia TTS - Text-to-Speech Model

Fast inference utilities:
    from dia.fast_inference import (
        create_fast_generate,      # GPU optimized
        create_cpu_generate,       # CPU optimized
        optimize_for_cpu,          # CPU configuration
    )
"""

from .model import Dia
from .config import DiaConfig

__all__ = ["Dia", "DiaConfig"]
