"""
High-level analysis and plotting for transient events.
"""

from .analysis import (
    analyze_thresholds,
    calculate_initial_background,
    calculate_smoothing_parameters,
    configure_logging,
    create_oscilloscope_plot,
    get_final_events,
    initialize_state,
    process_chunk,
    process_file,
)
from .event_detector import detect_events, merge_overlapping_events
from .event_plotter import EventPlotter
from pywf import get_waveform_params, rd, rd_chunked

__all__ = [
    "analyze_thresholds",
    "calculate_initial_background",
    "calculate_smoothing_parameters",
    "configure_logging",
    "create_oscilloscope_plot",
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
