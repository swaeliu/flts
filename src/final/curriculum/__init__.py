"""ReLoRA curriculum: training with per-client patience and Adam moment resets.

Public API
----------
    ReLoRAConfig          – dataclass extending the base Config with relora_* keys
    ClientPatienceTracker – per-client patience counter and convergence check
    prune_adam_moments    – magnitude-based partial zeroing of Adam m/v buffers
    CurriculumScheduler   – round-boundary orchestration (checkpoint, reset, warmup)
"""

from .config import ReLoRAConfig
from .patience import ClientPatienceTracker
from .adam_reset import prune_adam_moments
from .curriculum import CurriculumScheduler

__all__ = [
    "ReLoRAConfig",
    "ClientPatienceTracker",
    "prune_adam_moments",
    "CurriculumScheduler",
]
