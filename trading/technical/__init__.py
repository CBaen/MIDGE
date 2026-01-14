"""
Technical Analysis Module for MIDGE

Provides indicator calculations and signal generation for trading intelligence.
"""

from .indicators import TechnicalIndicators
from .signals import SignalGenerator

__all__ = ["TechnicalIndicators", "SignalGenerator"]
