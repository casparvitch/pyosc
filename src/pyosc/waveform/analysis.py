import base64
import io
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numba import njit
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import savgol_filter

from pyosc.oscplot.plot import OscilloscopePlot
from pyosc.waveform.event_detector import (
    MEDIAN_TO_STD_FACTOR,
    detect_events,
    merge_overlapping_events,
)
from pyosc.waveform.event_plotter import EventPlotter
from pyosc.waveform.io import rd
import xml.etree.ElementTree as ET


@njit
def _create_event_mask_numba(t: np.ndarray, events: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask to exclude event regions using numba for speed.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    events : np.ndarray
        Events array with shape (n_events, 2) where each row is [t_start, t_end].

    Returns
    -------
    np.ndarray
        Boolean mask where True means keep the sample, False means exclude.
    """
    mask = np.ones(len(t), dtype=np.bool_)

    for i in range(len(events)):
        t_start = events[i, 0]
        t_end = events[i, 1]

        # Find indices where time is within event bounds
        for j in range(len(t)):
            if t_start <= t[j] < t_end:
                mask[j] = False

    return mask


def extract_preview_image(xml_path: str, output_path: str) -> Optional[str]:
    """
    Extract preview image from XML sidecar and save as PNG.
    
    Parameters
    ----------
    xml_path : str
        Path to the XML sidecar file.
    output_path : str
        Path where to save the PNG file.
        
    Returns
    -------
    Optional[str]
        Path to saved PNG file, or None if no image found.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        logger.critical("Attempting to extract preview image from XML sidecar")
        
        # Find PreviewImage element
        preview_elem = root.find(".//PreviewImage")
        if preview_elem is None:
            logger.warning(f"No PreviewImage found in {xml_path}")
            return None
            
        image_data = preview_elem.get("ImageData")
        if not image_data:
            logger.warning(f"Empty ImageData in PreviewImage from {xml_path}")
            return None
            
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        
        # Open with PIL and save as PNG
        image = Image.open(io.BytesIO(image_bytes))
        image.save(output_path, "PNG")
        
        logger.info(f"Saved preview image: {output_path}")
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to extract preview image from {xml_path}: {e}")
        return None


def plot_preview_image(image_path: str, title: str = "Preview Image") -> None:
    """
    Display preview image using matplotlib.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
    title : str
        Title for the plot.
    """
    try:
        image = Image.open(image_path)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')  # Hide axes for cleaner display
        plt.show()  # Display the figure

    except Exception as e:
        logger.warning(f"Failed to display preview image {image_path}: {e}")


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure loguru logging with specified level.

    Parameters
    ----------
    log_level : str, default="INFO"
        Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


def load_data(
    name: str,
    sampling_interval: float,
    data_path: str,
    sidecar: Optional[str] = None,
    crop: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 1: Load waveform data from file.

    Parameters
    ----------
    name : str
        Filename of the waveform data.
    sampling_interval : float
        Sampling interval in seconds.
    data_path : str
        Path to data directory.
    sidecar : str, optional
        XML sidecar filename.
    crop : List[int], optional
        Crop indices [start, end].

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Time and signal arrays.
    """
    logger.info(f"Loading data from {name}")
    t, x = rd(
        name,
        sampling_interval,
        data_path=data_path,
        xml_filename=sidecar,
        crop=crop,
    )

    logger.debug(
        f"Signal statistics: min={np.min(x):.3g}, max={np.max(x):.3g}, mean={np.mean(x):.3g}, std={np.std(x):.3g}"
    )

    return t, x


def calculate_smoothing_parameters(
    sampling_interval: float,
    smooth_win_t: Optional[float],
    smooth_win_f: Optional[float],
    min_event_t: float,
    detection_snr: float,
    min_event_keep_snr: float,
    widen_frac: float,
    signal_polarity: int,
) -> Tuple[int, int]:
    """
    Calculate smoothing window size and minimum event length in samples.

    Parameters
    ----------
    sampling_interval : float
        Sampling interval in seconds.
    smooth_win_t : Optional[float]
        Smoothing window in seconds.
    smooth_win_f : Optional[float]
        Smoothing window in Hz.
    min_event_t : float
        Minimum event duration in seconds.
    detection_snr : float
        Detection SNR threshold.
    min_event_keep_snr : float
        Minimum event keep SNR threshold.
    widen_frac : float
        Fraction to widen detected events.
    signal_polarity : int
        Signal polarity (-1 for negative, +1 for positive).

    Returns
    -------
    Tuple[int, int]
        Smoothing window size and minimum event length in samples.
    """
    if smooth_win_t is not None:
        smooth_n = int(smooth_win_t / sampling_interval)
    elif smooth_win_f is not None:
        smooth_n = int(1 / (smooth_win_f * sampling_interval))
    else:
        raise ValueError("Set either smooth_win_t or smooth_win_f")

    if smooth_n % 2 == 0:
        smooth_n += 1

    min_event_n = int(min_event_t / sampling_interval)

    smooth_freq_hz = 1 / (smooth_n * sampling_interval)
    logger.info(
        f"--Smooth window: {smooth_n} samples ({smooth_win_t * 1e6:.1f} µs, {smooth_freq_hz:.1f} Hz)"
    )
    logger.info(
        f"--Min event length: {min_event_n} samples ({min_event_t * 1e6:.1f} µs)"
    )
    logger.info(f"--Detection SNR: {detection_snr}")
    logger.info(f"--Min keep SNR: {min_event_keep_snr}")
    logger.info(f"--Widen fraction: {widen_frac}")
    logger.info(
        f"--Signal polarity: {signal_polarity} ({'negative' if signal_polarity < 0 else 'positive'} events)"
    )

    return smooth_n, min_event_n


def calculate_initial_background(
    t: np.ndarray, x: np.ndarray, smooth_n: int, filter_type: str = "gaussian"
) -> np.ndarray:
    """
    Stage 2: Calculate initial background estimate.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Signal array.
    smooth_n : int
        Smoothing window size in samples.
    filter_type : str, default="gaussian"
        Filter type: "savgol", "gaussian", "moving_average", "median".

    Returns
    -------
    np.ndarray
        Initial background estimate.

    Notes
    -----
     1 Start with Gaussian - Best balance of speed, noise rejection, and event preservation
     2 Try Median if you see frequent spikes/glitches in your data
     3 Use Moving Average for maximum speed if events are well above noise
     4 Reserve Savitzky-Golay for final high-quality analysis of interesting datasets
    """
    logger.info(f"Calculating initial background using {filter_type} filter")

    if filter_type == "savgol":
        bg_initial = savgol_filter(x, smooth_n, 3).astype(np.float32)
    elif filter_type == "gaussian":
        sigma = smooth_n / 6.0  # Convert window to sigma (6-sigma rule)
        bg_initial = gaussian_filter1d(x.astype(np.float64), sigma).astype(np.float32)
    elif filter_type == "moving_average":
        # Use scipy's uniform_filter1d for proper edge handling
        bg_initial = uniform_filter1d(
            x.astype(np.float64), size=smooth_n, mode="nearest"
        ).astype(np.float32)
    elif filter_type == "median":
        bg_initial = median_filter(x.astype(np.float64), size=smooth_n).astype(
            np.float32
        )
    else:
        raise ValueError(
            f"Unknown filter_type: {filter_type}. Choose from 'savgol', 'gaussian', 'moving_average', 'median'"
        )

    logger.debug(
        f"Initial background: mean={np.mean(bg_initial):.3g}, std={np.std(bg_initial):.3g}"
    )
    return bg_initial


def estimate_noise(x: np.ndarray, bg_initial: np.ndarray) -> np.float32:
    """
    Stage 2: Estimate noise level.

    Parameters
    ----------
    x : np.ndarray
        Signal array.
    bg_initial : np.ndarray
        Initial background estimate.

    Returns
    -------
    np.float32
        Estimated noise level.
    """
    global_noise = np.float32(np.median(np.abs(x - bg_initial)) * MEDIAN_TO_STD_FACTOR)

    signal_rms = np.sqrt(np.mean(x**2))
    signal_range = np.max(x) - np.min(x)
    noise_pct_rms = 100 * global_noise / signal_rms if signal_rms > 0 else 0
    noise_pct_range = 100 * global_noise / signal_range if signal_range > 0 else 0

    logger.info(
        f"Global noise level: {global_noise:.3g} ({noise_pct_rms:.1f}% of RMS, {noise_pct_range:.1f}% of range)"
    )

    snr_estimate = np.std(x) / global_noise
    logger.info(f"Estimated signal SNR: {snr_estimate:.2f}")

    return global_noise


def detect_initial_events(
    t: np.ndarray,
    x: np.ndarray,
    bg_initial: np.ndarray,
    global_noise: np.float32,
    detection_snr: float,
    min_event_keep_snr: float,
    widen_frac: float,
    signal_polarity: int,
    min_event_n: int,
) -> np.ndarray:
    """
    Stage 3: Detect initial events.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Signal array.
    bg_initial : np.ndarray
        Initial background estimate.
    global_noise : np.float32
        Estimated noise level.
    detection_snr : float
        Detection SNR threshold.
    min_event_keep_snr : float
        Minimum event keep SNR threshold.
    widen_frac : float
        Fraction to widen detected events.
    signal_polarity : int
        Signal polarity (-1 for negative, +1 for positive).
    min_event_n : int
        Minimum event length in samples.

    Returns
    -------
    np.ndarray
        Array of initial events.
    """
    logger.info("Detecting initial events")
    min_event_amp = np.float32(min_event_keep_snr) * global_noise

    logger.info(f"Detection threshold: {detection_snr}σ below background")
    logger.info(f"Keep threshold: {min_event_keep_snr}σ below background")
    logger.info(f"Min event amplitude threshold: {min_event_amp:.3g}")

    events_initial, _ = detect_events(
        t,
        x,
        bg_initial,
        snr_threshold=np.float32(detection_snr),
        min_event_len=min_event_n,
        min_event_amp=min_event_amp,
        widen_frac=np.float32(widen_frac),
        global_noise=global_noise,
        signal_polarity=signal_polarity,
    )

    logger.info(f"Found {len(events_initial)} initial events after filtering")

    events_initial = merge_overlapping_events(events_initial)
    logger.info(f"After merging: {len(events_initial)} events")

    return events_initial


def calculate_clean_background(
    t: np.ndarray,
    x: np.ndarray,
    events_initial: np.ndarray,
    smooth_n: int,
    bg_initial: np.ndarray,
    filter_type: str = "gaussian",
    filter_order: int = 2,
) -> np.ndarray:
    """
    Stage 4: Calculate clean background by masking events.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Signal array.
    events_initial : np.ndarray
        Initial events array.
    smooth_n : int
        Smoothing window size in samples.
    bg_initial : np.ndarray
        Initial background estimate.
    filter_type : str, default="gaussian"
        Filter type: "savgol", "gaussian", "moving_average", "median".
    filter_order : int, default=2
        Order of the Savitzky-Golay filter (only used for filter_type="savgol").

    Returns
    -------
    np.ndarray
        Clean background estimate.


    Notes
    -----
     1 Start with Gaussian - Best balance of speed, noise rejection, and event preservation
     2 Try Median if you see frequent spikes/glitches in your data
     3 Use Moving Average for maximum speed if events are well above noise
     4 Reserve Savitzky-Golay for final high-quality analysis of interesting datasets
    """
    logger.info(f"Calculating clean background using {filter_type} filter")
    start_time = time.time()

    # Fast masking with numba
    mask = _create_event_mask_numba(t, events_initial)
    mask_time = time.time()

    logger.debug(
        f"Masked {np.sum(~mask)} samples ({100 * np.sum(~mask) / len(mask):.1f}%) for clean background"
    )

    t_masked = t[mask]
    x_masked = x[mask]

    if np.sum(mask) > 2 * smooth_n:
        # Check if we need interpolation (events detected and masking applied)
        if len(events_initial) == 0 or np.all(mask):
            # No events detected or no masking needed - skip interpolation
            logger.debug("No events to mask - using direct filtering")
            interp_start = time.time()
            x_interp = x
            interp_end = time.time()
        else:
            # Events detected - need interpolation
            interp_start = time.time()
            x_interp = np.interp(t, t_masked, x_masked)
            interp_end = time.time()

        filter_start = time.time()
        if filter_type == "savgol":
            bg_clean = savgol_filter(x_interp, smooth_n, filter_order).astype(
                np.float32
            )
        elif filter_type == "gaussian":
            sigma = smooth_n / 6.0  # Convert window to sigma (6-sigma rule)
            bg_clean = gaussian_filter1d(x_interp.astype(np.float64), sigma).astype(
                np.float32
            )
        elif filter_type == "moving_average":
            bg_clean = uniform_filter1d(
                x_interp.astype(np.float64), size=smooth_n, mode="nearest"
            ).astype(np.float32)
        elif filter_type == "median":
            bg_clean = median_filter(x_interp.astype(np.float64), size=smooth_n).astype(
                np.float32
            )
        else:
            raise ValueError(
                f"Unknown filter_type: {filter_type}. Choose from 'savgol', 'gaussian', 'moving_average', 'median'"
            )
        filter_end = time.time()

        logger.success(
            f"Timing: mask={mask_time - start_time:.3f}s, interp={interp_end - interp_start:.3f}s, filter={filter_end - filter_start:.3f}s"
        )
        logger.debug(
            f"Clean background: mean={np.mean(bg_clean):.3g}, std={np.std(bg_clean):.3g}"
        )
    else:
        logger.debug(
            "Insufficient unmasked samples for clean background - using initial"
        )
        bg_clean = bg_initial

    return bg_clean


def analyze_thresholds(
    x: np.ndarray,
    bg_clean: np.ndarray,
    global_noise: np.float32,
    detection_snr: float,
    min_event_keep_snr: float,
    signal_polarity: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze threshold statistics and create threshold arrays.

    Parameters
    ----------
    x : np.ndarray
        Signal array.
    bg_clean : np.ndarray
        Clean background estimate.
    global_noise : np.float32
        Estimated noise level.
    detection_snr : float
        Detection SNR threshold.
    min_event_keep_snr : float
        Minimum event keep SNR threshold.
    signal_polarity : int
        Signal polarity (-1 for negative, +1 for positive).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Detection and keep threshold arrays.
    """
    logger.info("Analyzing thresholds")

    if signal_polarity < 0:
        detection_threshold = bg_clean - detection_snr * global_noise
        keep_threshold = bg_clean - min_event_keep_snr * global_noise
        below_detection_pct = 100 * np.sum(x < detection_threshold) / len(x)
        below_keep_pct = 100 * np.sum(x < keep_threshold) / len(x)
        logger.info(f"Samples below detection threshold: {below_detection_pct:.2f}%")
        logger.info(f"Samples below keep threshold: {below_keep_pct:.2f}%")
    else:
        detection_threshold = bg_clean + detection_snr * global_noise
        keep_threshold = bg_clean + min_event_keep_snr * global_noise
        above_detection_pct = 100 * np.sum(x > detection_threshold) / len(x)
        above_keep_pct = 100 * np.sum(x > keep_threshold) / len(x)
        logger.info(f"Samples above detection threshold: {above_detection_pct:.2f}%")
        logger.info(f"Samples above keep threshold: {above_keep_pct:.2f}%")

    return detection_threshold, keep_threshold


def detect_final_events(
    t: np.ndarray,
    x: np.ndarray,
    bg_clean: np.ndarray,
    global_noise: np.float32,
    detection_snr: float,
    min_event_keep_snr: float,
    widen_frac: float,
    signal_polarity: int,
    min_event_n: int,
) -> np.ndarray:
    """
    Stage 5: Detect final events using clean background.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Signal array.
    bg_clean : np.ndarray
        Clean background estimate.
    global_noise : np.float32
        Estimated noise level.
    detection_snr : float
        Detection SNR threshold.
    min_event_keep_snr : float
        Minimum event keep SNR threshold.
    widen_frac : float
        Fraction to widen detected events.
    signal_polarity : int
        Signal polarity (-1 for negative, +1 for positive).
    min_event_n : int
        Minimum event length in samples.

    Returns
    -------
    np.ndarray
        Array of final events.
    """
    logger.info("Detecting final events")
    min_event_amp = np.float32(min_event_keep_snr) * global_noise

    events, noise = detect_events(
        t,
        x,
        bg_clean,
        snr_threshold=np.float32(detection_snr),
        min_event_len=min_event_n,
        min_event_amp=min_event_amp,
        widen_frac=np.float32(widen_frac),
        global_noise=global_noise,
        signal_polarity=signal_polarity,
    )

    events = merge_overlapping_events(events)
    logger.info(f"Detected {len(events)} final events")

    return events


def analyze_events(
    t: np.ndarray,
    x: np.ndarray,
    bg_clean: np.ndarray,
    events: np.ndarray,
    global_noise: np.float32,
    signal_polarity: int,
) -> None:
    """
    Stage 7: Analyze event characteristics.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Signal array.
    bg_clean : np.ndarray
        Clean background estimate.
    events : np.ndarray
        Events array.
    global_noise : np.float32
        Estimated noise level.
    signal_polarity : int
        Signal polarity (-1 for negative, +1 for positive).
    """
    if len(events) == 0:
        logger.info("No events to analyze")
        return
    if len(events) > 1000:
        logger.warning(
            f"Detected {len(events)} events, which is more than 1000. Skipping analysis."
        )
        return

    event_durations = (events[:, 1] - events[:, 0]) * 1000000  # Convert to µs
    event_amplitudes = []

    for t_start, t_end in events:
        event_mask = (t >= t_start) & (t < t_end)
        if np.any(event_mask):
            if signal_polarity < 0:
                amp = np.min(x[event_mask] - bg_clean[event_mask])
            else:
                amp = np.max(x[event_mask] - bg_clean[event_mask])
            event_amplitudes.append(abs(amp))

    if event_amplitudes:
        logger.info(
            f"Event durations (µs): min={np.min(event_durations):.2f}, max={np.max(event_durations):.2f}, mean={np.mean(event_durations):.2f}"
        )
        logger.info(
            f"Event amplitudes: min={np.min(event_amplitudes):.3g}, max={np.max(event_amplitudes):.3g}, mean={np.mean(event_amplitudes):.3g}"
        )
        logger.info(
            f"Event amplitude SNRs: min={np.min(event_amplitudes) / global_noise:.2f}, max={np.max(event_amplitudes) / global_noise:.2f}"
        )

    final_signal_rms = np.sqrt(np.mean(x**2))
    final_noise_pct_rms = (
        100 * global_noise / final_signal_rms if final_signal_rms > 0 else 0
    )
    final_signal_range = np.max(x) - np.min(x)
    final_noise_pct_range = (
        100 * global_noise / final_signal_range if final_signal_range > 0 else 0
    )

    logger.info(
        f"Noise summary: {global_noise:.3g} ({final_noise_pct_rms:.1f}% of RMS, {final_noise_pct_range:.1f}% of range)"
    )


def create_oscilloscope_plot(
    t: np.ndarray,
    x: np.ndarray,
    bg_initial: np.ndarray,
    bg_clean: np.ndarray,
    events: np.ndarray,
    detection_threshold: np.ndarray,
    keep_threshold: np.ndarray,
    name: str,
    detection_snr: float,
    min_event_keep_snr: float,
    max_plot_points: int,
    envelope_mode_limit: float,
    smooth_n: int,
    global_noise: Optional[np.float32] = None,
) -> OscilloscopePlot:
    """
    Stage 6: Create oscilloscope plot with all visualization elements.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Signal array.
    bg_initial : np.ndarray
        Initial background estimate.
    bg_clean : np.ndarray
        Clean background estimate.
    events : np.ndarray
        Events array.
    detection_threshold : np.ndarray
        Detection threshold array.
    keep_threshold : np.ndarray
        Keep threshold array.
    name : str
        Name for the plot.
    detection_snr : float
        Detection SNR threshold.
    min_event_keep_snr : float
        Minimum event keep SNR threshold.
    max_plot_points : int
        Maximum plot points for decimation.
    envelope_mode_limit : float
        Envelope mode limit.
    smooth_n : int
        Smoothing window size in samples.
    global_noise : Optional[np.float32], default=None
        Estimated noise level. If provided, will be plotted as a ribbon.

    Returns
    -------
    OscilloscopePlot
        Configured oscilloscope plot.
    """
    logger.info("Creating visualization")

    plot_name = name
    if global_noise is not None:
        plot_signal_rms = np.sqrt(np.mean(x**2))
        plot_noise_pct_rms = (
            100 * global_noise / plot_signal_rms if plot_signal_rms > 0 else 0
        )
        plot_name = f"{name} | Global noise: {global_noise:.3g} ({plot_noise_pct_rms:.1f}% of RMS)"

    plot = OscilloscopePlot(
        t,
        x,
        name=plot_name,
        max_plot_points=max_plot_points,
        mode_switch_threshold=envelope_mode_limit,
        envelope_window_samples=None,  # Envelope window now calculated automatically based on zoom
    )

    plot.add_line(
        t,
        bg_clean,
        label="Background",
        color="orange",
        alpha=0.6,
        linewidth=1.5,
        display_mode=OscilloscopePlot.MODE_BOTH,
    )

    if global_noise is not None:
        plot.add_ribbon(
            t,
            bg_clean,
            global_noise,
            label="Noise (±1σ)",
            color="gray",
            alpha=0.3,
            display_mode=OscilloscopePlot.MODE_DETAIL,
        )

    plot.add_line(
        t,
        detection_threshold,
        label=f"Detection ({detection_snr}σ)",
        color="red",
        alpha=0.7,
        linestyle=":",
        linewidth=1.5,
        display_mode=OscilloscopePlot.MODE_DETAIL,
    )

    plot.add_line(
        t,
        keep_threshold,
        label=f"Keep ({min_event_keep_snr}σ)",
        color="darkred",
        alpha=0.7,
        linestyle="--",
        linewidth=1.5,
        display_mode=OscilloscopePlot.MODE_DETAIL,
    )

    if len(events) > 0:
        plot.add_regions(
            events,
            label="Events",
            color="crimson",
            alpha=0.4,
            display_mode=OscilloscopePlot.MODE_BOTH,
        )

    plot.render()
    return plot


def process_file(
    name: str,
    sampling_interval: float,
    data_path: str,
    smooth_win_t: Optional[float] = None,
    smooth_win_f: Optional[float] = None,
    detection_snr: float = 3.0,
    min_event_keep_snr: float = 6.0,
    min_event_t: float = 0.75e-6,
    widen_frac: float = 10.0,
    signal_polarity: int = -1,
    max_plot_points: int = 10000,
    envelope_mode_limit: float = 10e-3,
    sidecar: Optional[str] = None,
    crop: Optional[List[int]] = None,
    yscale_mode: str = "snr",
    show_plots: bool = True,
    filter_type: str = "gaussian",
    filter_order: int = 2,
) -> None:
    """
    Process a single waveform file for event detection.

    Parameters
    ----------
    name : str
        Filename of the waveform data.
    sampling_interval : float
        Sampling interval in seconds.
    data_path : str
        Path to data directory.
    smooth_win_t : Optional[float], default=None
        Smoothing window in seconds.
    smooth_win_f : Optional[float], default=None
        Smoothing window in Hz.
    detection_snr : float, default=3.0
        Detection SNR threshold.
    min_event_keep_snr : float, default=6.0
        Minimum event keep SNR threshold.
    min_event_t : float, default=0.75e-6
        Minimum event duration in seconds.
    widen_frac : float, default=10.0
        Fraction to widen detected events.
    signal_polarity : int, default=-1
        Signal polarity (-1 for negative, +1 for positive).
    max_plot_points : int, default=10000
        Maximum plot points for decimation.
    envelope_mode_limit : float, default=10e-3
        Envelope mode limit.
    sidecar : str, optional
        XML sidecar filename.
    crop : List[int], optional
        Crop indices [start, end].
    yscale_mode : str, default="snr"
        Y-axis scaling mode for event plotter.
    show_plots : bool, default=True
        Whether to show plots interactively.
    filter_type : str, default="gaussian"
        Filter type for background smoothing: "savgol", "gaussian", "moving_average", "median".
    filter_order : int, default=2
        Order of the Savitzky-Golay filter (only used for filter_type="savgol").
    """
    start_time = time.time()
    logger.info(f"Processing {name} with parameters:")

    # Stage 1: Load Data
    t, x = load_data(name, sampling_interval, data_path, sidecar, crop)

    analysis_dir = data_path[:-1] if data_path.endswith("/") else data_path
    analysis_dir += "_analysis/"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # Extract and save preview image
    from pyosc.waveform.io import _get_xml_sidecar_path
    sidecar_path = _get_xml_sidecar_path(name, data_path, sidecar)
    logger.info(f"Attempting to extract preview from: {sidecar_path}")
    preview_path = os.path.join(analysis_dir, f"{name}_preview.png")
    saved_preview = extract_preview_image(sidecar_path, preview_path)

    if saved_preview and show_plots:
        plot_preview_image(saved_preview, f"Preview: {name}")

    # Calculate parameters
    smooth_n, min_event_n = calculate_smoothing_parameters(
        sampling_interval,
        smooth_win_t,
        smooth_win_f,
        min_event_t,
        detection_snr,
        min_event_keep_snr,
        widen_frac,
        signal_polarity,
    )

    # Stage 2: Background Calculation
    bg_initial = calculate_initial_background(t, x, smooth_n, filter_type)
    global_noise = estimate_noise(x, bg_initial)

    # Stage 3: Initial Event Detection
    bg_time = time.time()
    logger.debug(f"Background calculation took {bg_time - start_time:.3f}s")
    events_initial = detect_initial_events(
        t,
        x,
        bg_initial,
        global_noise,
        detection_snr,
        min_event_keep_snr,
        widen_frac,
        signal_polarity,
        min_event_n,
    )

    # Stage 4: Clean Background Calculation
    bg_clean = calculate_clean_background(
        t, x, events_initial, smooth_n, bg_initial, filter_type, filter_order
    )

    # Analyze thresholds
    detection_threshold, keep_threshold = analyze_thresholds(
        x, bg_clean, global_noise, detection_snr, min_event_keep_snr, signal_polarity
    )

    # Stage 5: Final Event Detection
    events = detect_final_events(
        t,
        x,
        bg_clean,
        global_noise,
        detection_snr,
        min_event_keep_snr,
        widen_frac,
        signal_polarity,
        min_event_n,
    )
    detection_time = time.time()
    logger.debug(f"Event detection took {detection_time - bg_time:.3f}s")

    # Stage 7: Event Analysis
    analyze_events(t, x, bg_clean, events, global_noise, signal_polarity)

    logger.debug(f"Total processing time: {detection_time - start_time:.3f}s")

    # Stage 6: Visualization
    plot = create_oscilloscope_plot(
        t,
        x,
        bg_initial,
        bg_clean,
        events,
        detection_threshold,
        keep_threshold,
        name,
        detection_snr,
        min_event_keep_snr,
        max_plot_points,
        envelope_mode_limit,
        smooth_n,
        global_noise=global_noise,
    )

    # Save plots

    plot.save(analysis_dir + f"{name}_trace.png")

    # Create event plotter
    event_plotter = EventPlotter(
        plot,
        events,
        bg_clean=bg_clean,
        global_noise=global_noise,
        y_scale_mode=yscale_mode,
    )
    event_plotter.plot_events_grid(max_events=16)
    event_plotter.save(analysis_dir + f"{name}_events.png")

    if show_plots:
        plt.show(block=True)
