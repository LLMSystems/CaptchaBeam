"""Selectors: choose one final answer across preprocessing variants."""
from .agreement import AgreementSelector
from .base import Selector
from .confidence import ConfidenceSelector

__all__ = ["Selector", "AgreementSelector", "ConfidenceSelector"]
