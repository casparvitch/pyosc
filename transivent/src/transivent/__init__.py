"""
High-level analysis and plotting for transient events.
"""

from .analysis import (
    calculate_initial_background,
    calculate_smoothing_parameters,
    configure_logging,
    get_final_events,
    initialize_state,
    process_chunk,
    process_file,
)
from .event_detector import detect_events, merge_overlapping_events
from .event_plotter import EventPlotter
from pywf import get_waveform_params, rd, rd_chunked

__all__ = [
    "calculate_initial_background",
    "calculate_smoothing_parameters",
    "configure_logging",
    "detect_events",
    "EventPlotter",
    "get_final_events",
    "get_waveform_params",
    "initialize_state",
    "merge_overlapping_events",
    "process_chunk",
    "process_file",
    "rd",
    "rd_chunked",
]
