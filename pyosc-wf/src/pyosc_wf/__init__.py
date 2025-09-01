"""
Waveform-specific components for PyOSC.

This package contains components specific to waveform data processing and visualization.
"""

from pyosc.waveform.analysis import configure_logging, process_file
from pyosc.waveform.event_detector import detect_events, merge_overlapping_events
from pyosc.waveform.event_plotter import EventPlotter
from pyosc.waveform.io import get_waveform_params, rd

__all__ = [
    "get_waveform_params",
    "rd",
    "detect_events",
    "merge_overlapping_events",
    "EventPlotter",
    "configure_logging",
    "process_file",
]
