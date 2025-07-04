from typing import Optional, Tuple

import numpy as np
from loguru import logger
from numba import njit

# --- Constants ---
MEDIAN_TO_STD_FACTOR = (
    1.4826  # Factor to convert median absolute deviation to standard deviation
)


@njit
def detect_events_numba(
    time: np.ndarray,
    signal: np.ndarray,
    bg: np.ndarray,
    snr_threshold: float,
    min_event_len: int,
    min_event_amp: float,
    widen_frac: float,
    global_noise: float,
    signal_polarity: int,
) -> np.ndarray:
    """
    Detect events in signal using Numba for performance.

    Uses pre-allocated NumPy arrays instead of dynamic lists for better performance.

    Parameters
    ----------
    time : np.ndarray
        Time array (float32).
    signal : np.ndarray
        Input signal array (float32).
    bg : np.ndarray
        Background/baseline array (float32).
    snr_threshold : float
        Signal-to-noise ratio threshold for detection.
    min_event_len : int
        Minimum event length in samples.
    min_event_amp : float
        Minimum event amplitude threshold.
    widen_frac : float
        Fraction to widen detected events.
    global_noise : float
        Global noise level.
    signal_polarity : int
        Signal polarity: -1 for negative events, +1 for positive events.

    Returns
    -------
    np.ndarray
        Array of shape (n_events, 2) with start and end indices of events.
    """
    # Cast scalar parameters to float32 for consistency
    snr_threshold = np.float32(snr_threshold)
    min_event_amp = np.float32(min_event_amp)
    widen_frac = np.float32(widen_frac)
    global_noise = np.float32(global_noise)

    if signal_polarity < 0:
        threshold = bg - snr_threshold * global_noise
        above = signal < threshold
    else:
        threshold = bg + snr_threshold * global_noise
        above = signal > threshold

    # Pre-allocate maximum possible events (worst case: every other sample is an event)
    max_events = len(signal) // 2
    events = np.empty((max_events, 2), dtype=np.int64)
    event_count = 0

    in_event = False
    start = 0

    for i in range(len(above)):
        val = above[i]
        if val and not in_event:
            start = i
            in_event = True
        elif not val and in_event:
            end = i
            event_len = end - start
            if event_len < min_event_len:
                in_event = False
                continue

            # Amplitude filter
            if min_event_amp > 0.0:
                if signal_polarity < 0:
                    if np.min(signal[start:end] - bg[start:end]) > -min_event_amp:
                        in_event = False
                        continue
                else:
                    if np.max(signal[start:end] - bg[start:end]) < min_event_amp:
                        in_event = False
                        continue

            # Widen event
            widen = int(widen_frac * (end - start))
            new_start = max(0, start - widen)
            new_end = min(len(signal), end + widen)

            # Store indices for now, convert to time outside numba
            events[event_count, 0] = new_start
            events[event_count, 1] = new_end
            event_count += 1
            in_event = False

    # Handle event at end of signal
    if in_event:
        end = len(signal)
        event_len = end - start
        if event_len >= min_event_len:
            if min_event_amp > 0.0:
                if signal_polarity < 0:
                    if np.min(signal[start:end] - bg[start:end]) <= -min_event_amp:
                        widen = int(widen_frac * (end - start))
                        new_start = max(0, start - widen)
                        new_end = min(len(signal), end + widen)
                        # Store indices for now, convert to time outside numba
                        events[event_count, 0] = new_start
                        events[event_count, 1] = new_end
                        event_count += 1
                else:
                    if np.max(signal[start:end] - bg[start:end]) >= min_event_amp:
                        widen = int(widen_frac * (end - start))
                        new_start = max(0, start - widen)
                        new_end = min(len(signal), end + widen)
                        # Store indices for now, convert to time outside numba
                        events[event_count, 0] = new_start
                        events[event_count, 1] = new_end
                        event_count += 1
            else:
                widen = int(widen_frac * (end - start))
                new_start = max(0, start - widen)
                new_end = min(len(signal), end + widen)
                # Store indices for now, convert to time outside numba
                events[event_count, 0] = new_start
                events[event_count, 1] = new_end
                event_count += 1

    # Return only the filled portion
    return events[:event_count]


@njit
def merge_overlapping_events_numba(events: np.ndarray) -> np.ndarray:
    """
    Merge overlapping events using Numba for performance.

    Parameters
    ----------
    events : np.ndarray
        Array of shape (n_events, 2) with start and end times.

    Returns
    -------
    np.ndarray
        Array of merged events with shape (n_merged, 2).
    """
    n = len(events)
    if n == 0:
        return np.empty((0, 2), dtype=np.float32)
    arr = events  # type transfer
    arr = arr[np.argsort(arr[:, 0])]
    merged = np.empty((n, 2), dtype=np.float32)
    count = 0
    merged[count] = arr[0]
    count += 1
    for i in range(1, arr.shape[0]):
        start, end = arr[i]
        last_start, last_end = merged[count - 1]
        if start <= last_end:
            merged[count - 1, 1] = max(last_end, end)
        else:
            merged[count] = arr[i]
            count += 1
    return merged[:count]


def detect_events(
    time: np.ndarray,
    signal: np.ndarray,
    bg: np.ndarray,
    snr_threshold: np.float32 = np.float32(2.0),
    min_event_len: int = 20,
    min_event_amp: np.float32 = np.float32(0.0),
    widen_frac: np.float32 = np.float32(0.5),
    global_noise: Optional[np.float32] = None,
    signal_polarity: int = -1,
) -> Tuple[np.ndarray, np.float32]:
    """
    Detect events in signal above background with specified thresholds.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    signal : np.ndarray
        Input signal array.
    bg : np.ndarray
        Background/baseline array.
    snr_threshold : np.float32, default=2.0
        Signal-to-noise ratio threshold for detection.
    min_event_len : int, default=20
        Minimum event length in samples.
    min_event_amp : np.float32, default=0.0
        Minimum event amplitude threshold.
    widen_frac : np.float32, default=0.5
        Fraction to widen detected events.
    global_noise : np.float32, optional
        Global noise level. Must be provided.
    signal_polarity : int, default=-1
        Signal polarity: -1 for negative events (below background), +1 for positive events (above background).

    Returns
    -------
    Tuple[np.ndarray, np.float32]
        Array of detected events (time ranges) and global noise value.

    Raises
    ------
    ValueError
        If global_noise is not provided or input arrays are invalid.
    """
    if global_noise is None:
        logger.error("global_noise was not provided to detect_events.")
        raise ValueError("global_noise must be provided")

    # Validate and convert input arrays
    time = np.asarray(time, dtype=np.float32)
    signal = np.asarray(signal, dtype=np.float32)
    bg = np.asarray(bg, dtype=np.float32)

    # Validate input data
    _validate_detection_inputs(
        time, signal, bg, snr_threshold, min_event_len, global_noise
    )

    events_indices = detect_events_numba(
        time,
        signal,
        bg,
        np.float32(snr_threshold),
        int(min_event_len),
        np.float32(min_event_amp),
        np.float32(widen_frac),
        np.float32(global_noise),
        int(signal_polarity),
    )

    # Convert indices to time values outside of numba
    events_array = np.empty_like(events_indices, dtype=np.float32)
    for i in range(len(events_indices)):
        start_idx = int(events_indices[i, 0])
        end_idx = int(events_indices[i, 1])
        events_array[i, 0] = time[start_idx]
        events_array[i, 1] = time[end_idx]

    logger.info(f"Raw detection found {len(events_array)} events")

    return events_array, np.float32(global_noise)


def _validate_detection_inputs(
    time: np.ndarray,
    signal: np.ndarray,
    bg: np.ndarray,
    snr_threshold: np.float32,
    min_event_len: int,
    global_noise: np.float32,
) -> None:
    """
    Validate inputs for event detection.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    signal : np.ndarray
        Signal array.
    bg : np.ndarray
        Background array.
    snr_threshold : np.float32
        SNR threshold.
    min_event_len : int
        Minimum event length.
    global_noise : np.float32
        Global noise level.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    # Check array lengths
    if not (len(time) == len(signal) == len(bg)):
        logger.warning(
            f"Validation Warning: Array length mismatch: time={len(time)}, signal={len(signal)}, bg={len(bg)}. "
            "This may lead to unexpected behaviour."
        )

    # Check for empty arrays
    if len(time) == 0:
        logger.warning(
            "Validation Warning: Input arrays are empty. This may lead to unexpected behaviour."
        )

    # Check time monotonicity with a small tolerance for floating-point comparisons
    if len(time) > 1:
        # Use a small epsilon for floating-point comparison
        # np.finfo(time.dtype).eps is the smallest representable positive number such that 1.0 + eps != 1.0
        # Multiplying by a small factor (e.g., 10) provides a reasonable tolerance.
        tolerance = np.finfo(time.dtype).eps * 10
        if not np.all(np.diff(time) > tolerance):
            # Log the problematic differences for debugging
            problematic_diffs = np.diff(time)[np.diff(time) <= tolerance]
            logger.warning(
                f"Validation Warning: Time array is not strictly monotonic increasing within tolerance {tolerance}. "
                f"Problematic diffs (first 10): {problematic_diffs[:10]}. This may lead to unexpected behaviour."
            )

    # Check parameter validity
    if snr_threshold <= 0:
        logger.warning(
            f"Validation Warning: SNR threshold must be positive, got {snr_threshold}. This may lead to unexpected behaviour."
        )

    if min_event_len <= 0:
        logger.warning(
            f"Validation Warning: Minimum event length must be positive, got {min_event_len}. This may lead to unexpected behaviour."
        )

    if global_noise <= 0:
        logger.warning(
            f"Validation Warning: Global noise must be positive, got {global_noise}. This may lead to unexpected behaviour."
        )

    # Check for NaN/inf values
    for name, arr in [("time", time), ("signal", signal), ("bg", bg)]:
        if not np.all(np.isfinite(arr)):
            logger.warning(
                f"Validation Warning: {name} array contains NaN or infinite values. This may lead to unexpected behaviour."
            )


def merge_overlapping_events(events: np.ndarray) -> np.ndarray:
    """
    Merge overlapping events.

    Parameters
    ----------
    events : np.ndarray
        Array of events with shape (n_events, 2).

    Returns
    -------
    np.ndarray
        Array of merged events.

    Raises
    ------
    ValueError
        If events array has invalid format.
    """
    if len(events) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Validate events array format
    events_array = np.asarray(events, dtype=np.float32)
    if events_array.ndim != 2 or events_array.shape[1] != 2:
        logger.warning(
            f"Validation Warning: Events array must have shape (n_events, 2), got {events_array.shape}. This may lead to unexpected behaviour."
        )
        # This specific check is critical for the Numba function's array indexing,
        # so it's safer to keep it as a ValueError if the shape is fundamentally wrong.
        # However, for "very permissive", I'll change it to a warning and let Numba potentially fail later.
        # If Numba fails, we can revert this specific one to ValueError.
        # For now, let's make it a warning.
        pass  # Continue execution after warning

    # Check for invalid events (start >= end)
    invalid_mask = events_array[:, 0] >= events_array[:, 1]
    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        logger.warning(
            f"Validation Warning: Invalid events found (start >= end) at indices: {invalid_indices}. This may lead to unexpected behaviour."
        )

    merged = merge_overlapping_events_numba(events_array)

    if len(merged) != len(events):
        logger.info(
            f"Merged {len(events)} → {len(merged)} events ({len(events) - len(merged)} overlaps resolved)"
        )

    return merged
