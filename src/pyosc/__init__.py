"""
PyOSC: Python Oscilloscope Plotting Library

A library for visualizing and analyzing time-series data with oscilloscope-like features.
"""

# Import from oscplot subpackage
from pyosc.oscplot.coordinate_manager import CoordinateManager
from pyosc.oscplot.data_manager import TimeSeriesDataManager
from pyosc.oscplot.decimation import DecimationManager
from pyosc.oscplot.display_state import DisplayState
from pyosc.oscplot.plot import OscilloscopePlot
from pyosc.waveform.event_detector import detect_events, merge_overlapping_events
from pyosc.waveform.event_plotter import EventPlotter

# Import from waveform subpackage
from pyosc.waveform.io import get_waveform_params, rd

__all__ = [
    # General oscilloscope plotting
    "OscilloscopePlot",
    "TimeSeriesDataManager",
    "CoordinateManager",
    "DisplayState",
    "DecimationManager",
    # Waveform-specific components
    "get_waveform_params",
    "rd",
    "detect_events",
    "merge_overlapping_events",
    "EventPlotter",
]
