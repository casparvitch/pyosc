"""
General-purpose oscilloscope plotting components for PyOSC.

This package contains the core plotting functionality that can be used
with any time-series data, not just waveforms.
"""

from .coordinate_manager import CoordinateManager
from .data_manager import TimeSeriesDataManager
from .decimation import DecimationManager
from .display_state import DisplayState
from .plot import OscilloscopePlot

__all__ = [
    "OscilloscopePlot",
    "TimeSeriesDataManager",
    "CoordinateManager",
    "DisplayState",
    "DecimationManager",
]
