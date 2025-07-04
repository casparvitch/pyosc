"""
General-purpose oscilloscope plotting components for PyOSC.

This package contains the core plotting functionality that can be used
with any time-series data, not just waveforms.
"""

from pyosc.oscplot.coordinate_manager import CoordinateManager
from pyosc.oscplot.data_manager import TimeSeriesDataManager
from pyosc.oscplot.decimation import DecimationManager
from pyosc.oscplot.display_state import DisplayState
from pyosc.oscplot.plot import OscilloscopePlot

__all__ = [
    "OscilloscopePlot",
    "TimeSeriesDataManager",
    "CoordinateManager",
    "DisplayState",
    "DecimationManager",
]
