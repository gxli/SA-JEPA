"""sajepa — Scale-Aware Joint-Embedding Predictive Architecture for Physical Fields."""

from src.api import ScaleAwareJEPA
from src.utils.memory import OOMSafeTrainer, clear_memory_cache, compute_accumulation_steps

__all__ = ["ScaleAwareJEPA", "OOMSafeTrainer", "clear_memory_cache", "compute_accumulation_steps"]
