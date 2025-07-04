from typing import Optional, Tuple

import numpy as np
from loguru import logger
from matplotlib.ticker import FuncFormatter

# Time unit boundaries (hysteresis)
PICOSECOND_BOUNDARY = 0.8e-9
NANOSECOND_BOUNDARY = 0.8e-6
MICROSECOND_BOUNDARY = 0.8e-3
MILLISECOND_BOUNDARY = 0.8

# Offset thresholds
OFFSET_SPAN_MULTIPLIER = 10
OFFSET_TIME_THRESHOLD = 1e-3  # 1ms


def _get_optimal_time_unit_and_scale(
    time_array_or_span: np.ndarray | float,
) -> Tuple[str, np.float32]:
    """
    Determines the optimal time unit and scaling factor for a given time array or span.

    Uses hysteresis boundaries to prevent oscillation near unit boundaries.

    Parameters
    ----------
    time_array_or_span : np.ndarray | float
        A NumPy array representing time in seconds, or a single float representing a time span in seconds.

    Returns
    -------
    Tuple[str, np.float32]
        A tuple containing the time unit string (e.g., "s", "ms", "us", "ns")
        and the corresponding scaling factor (e.0, 1e3, 1e6, 1e9).
    """
    if isinstance(time_array_or_span, np.ndarray):
        # Handle empty array case to prevent errors
        if time_array_or_span.size == 0:
            return "s", np.float32(1.0)  # Default to seconds if no data
        max_val = np.max(time_array_or_span)
    else:  # Assume it's a float representing a span
        max_val = time_array_or_span

    # Use hysteresis boundaries to prevent oscillation near unit boundaries
    if max_val < PICOSECOND_BOUNDARY:
        return "ps", np.float32(1e12)
    elif max_val < NANOSECOND_BOUNDARY:
        return "ns", np.float32(1e9)
    elif max_val < MICROSECOND_BOUNDARY:
        return "us", np.float32(1e6)
    elif max_val < MILLISECOND_BOUNDARY:
        return "ms", np.float32(1e3)
    else:
        return "s", np.float32(1.0)


def _determine_offset_display_params(
    xlim_raw: Tuple[np.float32, np.float32], time_span_raw: np.float32
) -> Tuple[str, np.float32, Optional[np.float32], Optional[str]]:
    """
    Determine display parameters including offset for optimal readability.

    Parameters
    ----------
    xlim_raw : Tuple[np.float32, np.float32]
        Current x-axis limits in raw time (seconds).
    time_span_raw : np.float32
        Time span of current view in seconds.

    Returns
    -------
    Tuple[str, np.float32, Optional[np.float32], Optional[str]]
        Display unit, display scale, offset time (raw seconds), offset unit string.
        If no offset is needed, offset_time and offset_unit will be None.
    """
    # Get optimal unit for the time span
    display_unit, display_scale = _get_optimal_time_unit_and_scale(time_span_raw)

    # Determine if we need an offset
    # Use offset if the start time is significantly larger than the span
    xlim_start = xlim_raw[0]

    # Use offset if start time is more than threshold multiplier of the span, and span is small
    use_offset = (abs(xlim_start) > OFFSET_SPAN_MULTIPLIER * time_span_raw) and (
        time_span_raw < np.float32(OFFSET_TIME_THRESHOLD)
    )

    if use_offset:
        # Choose appropriate unit for the offset
        if abs(xlim_start) >= np.float32(1.0):  # >= 1 second
            offset_unit = "s"
            offset_scale = np.float32(1.0)
        elif abs(xlim_start) >= np.float32(1e-3):  # >= 1 millisecond
            offset_unit = "ms"
            offset_scale = np.float32(1e3)
        elif abs(xlim_start) >= np.float32(1e-6):  # >= 1 microsecond
            offset_unit = "us"
            offset_scale = np.float32(1e6)
        else:
            offset_unit = "ns"
            offset_scale = np.float32(1e9)

        return display_unit, display_scale, xlim_start, offset_unit
    else:
        return display_unit, display_scale, None, None


def _create_time_formatter(
    offset_time_raw: Optional[np.float32], display_scale: np.float32
) -> FuncFormatter:
    """
    Create a FuncFormatter for time axis tick labels.

    Parameters
    ----------
    offset_time_raw : Optional[np.float32]
        Offset time in raw seconds. If None, no offset is applied.
    display_scale : np.float32
        Scale factor for display units.

    Returns
    -------
    FuncFormatter
        Matplotlib formatter for tick labels.
    """

    def formatter(x, pos):
        # x is already in display units (relative to offset if applicable)
        # Format with appropriate precision based on scale
        if display_scale >= np.float32(1e9):  # nanoseconds or smaller
            return f"{x:.0f}"
        elif display_scale >= np.float32(1e6):  # microseconds
            return f"{x:.0f}"
        elif display_scale >= np.float32(1e3):  # milliseconds
            return f"{x:.1f}"
        else:  # seconds
            return f"{x:.3f}"

    return FuncFormatter(formatter)


class DisplayState:
    """
    Manages display state and mode switching logic.

    Centralises state management to reduce complexity and flag interactions.
    """

    def __init__(
        self,
        original_time_unit: str,
        original_time_scale: np.float32,
        envelope_limit: np.float32,
    ):
        """
        Initialise display state.

        Parameters
        ----------
        original_time_unit : str
            Original time unit string.
        original_time_scale : np.float32
            Original time scaling factor.
        envelope_limit : np.float32
            Time span threshold for envelope mode.
        """
        # Time scaling
        self.original_time_unit = original_time_unit
        self.original_time_scale = original_time_scale
        self.current_time_unit = original_time_unit
        self.current_time_scale = original_time_scale

        # Display mode
        self.current_mode: Optional[str] = None
        self.envelope_limit = envelope_limit

        # Offset parameters
        self.offset_time_raw: Optional[np.float32] = None
        self.offset_unit: Optional[str] = None

        # Single state flag - simplified
        self._updating = False

    def get_time_unit_and_scale(self, t: np.ndarray) -> Tuple[str, np.float32]:
        """
        Automatically select appropriate time unit and scale for plotting.

        Parameters
        ----------
        t : np.ndarray
            Time array.

        Returns
        -------
        Tuple[str, np.float32]
            Time unit string and scaling factor.
        """
        # Delegate to the new utility function
        return _get_optimal_time_unit_and_scale(t)

    def update_display_params(
        self, xlim_raw: Tuple[np.float32, np.float32], time_span_raw: np.float32
    ) -> bool:
        """
        Update display parameters including offset based on current view.

        Parameters
        ----------
        xlim_raw : Tuple[np.float32, np.float32]
            Current x-axis limits in raw time (seconds).
        time_span_raw : np.float32
            Time span of current view in seconds.

        Returns
        -------
        bool
            True if display parameters changed, False otherwise.
        """
        display_unit, display_scale, offset_time, offset_unit = (
            _determine_offset_display_params(xlim_raw, time_span_raw)
        )

        # Check if anything changed
        params_changed = (
            display_unit != self.current_time_unit
            or display_scale != self.current_time_scale
            or offset_time != self.offset_time_raw
            or offset_unit != self.offset_unit
        )

        if params_changed:
            logger.info(
                f"Display params changed: unit={display_unit}, scale={display_scale:.1e}, offset={offset_time}, offset_unit={offset_unit}"
            )
            self.current_time_unit = display_unit
            self.current_time_scale = display_scale
            self.offset_time_raw = offset_time
            self.offset_unit = offset_unit
            return True

        return False

    def should_use_envelope(self, time_span_raw: np.float32) -> bool:
        """Determine if envelope mode should be used based on time span."""
        return time_span_raw > self.envelope_limit

    def should_show_thresholds(self, time_span_raw: np.float32) -> bool:
        """Determine if threshold lines should be shown based on time span."""
        return time_span_raw < self.envelope_limit

    def update_time_scale(self, time_span_raw: np.float32) -> bool:
        """
        Update time scale based on current view span.

        Returns True if scale changed, False otherwise.
        """
        # Delegate to the new utility function for span
        new_unit, new_scale = _get_optimal_time_unit_and_scale(time_span_raw)

        if new_scale != self.current_time_scale:
            logger.info(
                f"Time scale changed from {self.current_time_unit} ({self.current_time_scale:.1e}) to {new_unit} ({new_scale:.1e})"
            )
            self.current_time_unit = new_unit
            self.current_time_scale = new_scale
            return True

        return False

    def reset_to_original_scale(self) -> None:
        """Reset time scale to original values."""
        self.current_time_unit = self.original_time_unit
        self.current_time_scale = self.original_time_scale
        logger.info(
            f"Reset to original scale: {self.current_time_unit} ({self.current_time_scale:.1e})"
        )

    def reset_to_initial_state(self) -> None:
        """Reset all display parameters to initial values."""
        self.current_time_unit = self.original_time_unit
        self.current_time_scale = self.original_time_scale
        self.offset_time_raw = None
        self.offset_unit = None
        self.current_mode = None
        self._updating = False

    def set_updating(self, value: bool = True) -> None:
        """Set updating state to prevent recursion."""
        self._updating = value

    def is_updating(self) -> bool:
        """Check if currently updating."""
        return self._updating
