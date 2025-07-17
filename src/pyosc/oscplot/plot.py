from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.ticker import MultipleLocator

from .coordinate_manager import CoordinateManager
from .data_manager import TimeSeriesDataManager
from .decimation import DecimationManager
from .display_state import (
    DisplayState,
    _create_time_formatter,
    _get_optimal_time_unit_and_scale,
)


class OscilloscopePlot:
    """
    General-purpose plotting class for time-series data with zoom and decimation.

    Uses separate managers for data, decimation, and state to reduce complexity.
    Supports different visualization elements (lines, envelopes, ribbons, regions)
    that can be displayed in different modes (envelope when zoomed out, detail when zoomed in).
    """

    # Mode constants
    MODE_ENVELOPE = 1  # Zoomed out mode
    MODE_DETAIL = 2  # Zoomed in mode
    MODE_BOTH = 3  # Both modes

    # Default styling constants
    DEFAULT_MAX_PLOT_POINTS = 10000
    DEFAULT_MODE_SWITCH_THRESHOLD = 10e-3  # 10 ms
    DEFAULT_MIN_Y_RANGE_DEFAULT = 1e-9  # Default minimum Y-axis range (e.g., 1 nV)
    DEFAULT_Y_MARGIN_FRACTION = 0.15
    DEFAULT_SIGNAL_LINE_WIDTH = 1.0
    DEFAULT_SIGNAL_ALPHA = 0.75
    DEFAULT_ENVELOPE_ALPHA = 0.4
    DEFAULT_REGION_ALPHA = 0.4
    DEFAULT_REGION_ZORDER = -5

    def __init__(
        self,
        t: Union[np.ndarray, List[np.ndarray]],
        x: Union[np.ndarray, List[np.ndarray]],
        name: Union[str, List[str]] = "Waveform",
        trace_colors: Optional[List[str]] = None,
        # Core display parameters
        max_plot_points: int = DEFAULT_MAX_PLOT_POINTS,
        mode_switch_threshold: float = DEFAULT_MODE_SWITCH_THRESHOLD,
        min_y_range: Optional[float] = None,  # New parameter for minimum Y-axis range
        y_margin_fraction: float = DEFAULT_Y_MARGIN_FRACTION,
        signal_line_width: float = DEFAULT_SIGNAL_LINE_WIDTH,
        signal_alpha: float = DEFAULT_SIGNAL_ALPHA,
        envelope_alpha: float = DEFAULT_ENVELOPE_ALPHA,
        region_alpha: float = DEFAULT_REGION_ALPHA,
        region_zorder: int = DEFAULT_REGION_ZORDER,
        envelope_window_samples: Optional[int] = None,
    ):
        """
        Initialize the OscilloscopePlot with time series data.

        Parameters
        ----------
        t : Union[np.ndarray, List[np.ndarray]]
            Time array(s) (raw time in seconds). Can be a single array shared by all traces
            or a list of arrays, one per trace.
        x : Union[np.ndarray, List[np.ndarray]]
            Signal array(s). If t is a single array, x can be a 2D array (traces x samples)
            or a list of 1D arrays. If t is a list, x must be a list of equal length.
        name : Union[str, List[str]], default="Waveform"
            Name(s) for plot title. Can be a single string or a list of strings.
        trace_colors : Optional[List[str]], default=None
            Colors for each trace. If None, default colors will be used.
        max_plot_points : int, default=10000
            Maximum number of points to display on the plot. Data will be decimated if it exceeds this.
        mode_switch_threshold : float, default=10e-3
            Time span (in seconds) above which the plot switches to envelope mode.
        min_y_range : Optional[float], default=None
            Minimum Y-axis range to enforce. If None, a default small value is used.
        y_margin_fraction : float, default=0.05
            Fraction of data range to add as margin to Y-axis limits.
        signal_line_width : float, default=1.0
            Line width for the raw signal plot.
        signal_alpha : float, default=0.75
            Alpha (transparency) for the raw signal plot.
        envelope_alpha : float, default=0.4
            Alpha (transparency) for the envelope fill.
        region_alpha : float, default=0.4
            Alpha (transparency) for region highlight fills.
        region_zorder : int, default=-5
            Z-order for region highlight fills (lower means further back).
        envelope_window_samples : Optional[int], default=None
            DEPRECATED: Window size in samples for envelope calculation.
            Envelope window is now calculated automatically based on max_plot_points and zoom level.
            This parameter is ignored but kept for backward compatibility.
        """
        # Store styling parameters directly as instance attributes
        self.max_plot_points = max_plot_points
        self.mode_switch_threshold = np.float32(mode_switch_threshold)
        self.min_y_range = (
            np.float32(min_y_range)
            if min_y_range is not None
            else self.DEFAULT_MIN_Y_RANGE_DEFAULT
        )
        self.y_margin_fraction = np.float32(y_margin_fraction)
        self.signal_line_width = signal_line_width
        self.signal_alpha = signal_alpha
        self.envelope_alpha = envelope_alpha
        self.region_alpha = region_alpha
        self.region_zorder = region_zorder
        # envelope_window_samples is now deprecated - envelope window is calculated automatically
        # Keep the parameter for backward compatibility but don't use it
        if envelope_window_samples is not None:
            logger.warning("envelope_window_samples parameter is deprecated. Envelope window is now calculated automatically based on zoom level.")

        # Initialize managers
        self.data = TimeSeriesDataManager(t, x, name, trace_colors)
        self.decimator = DecimationManager()

        # Pre-decimate main signal data for envelope view
        for i in range(self.data.num_traces):
            self.decimator.pre_decimate_data(
                data_id=i,  # Use trace_idx as data_id
                t=self.data.t_arrays[i],
                x=self.data.x_arrays[i],
                max_points=self.max_plot_points,
                envelope_window_samples=None,  # Envelope window calculated automatically
            )

        # Initialize display state using the first trace's time array
        initial_time_unit, initial_time_scale = _get_optimal_time_unit_and_scale(
            self.data.t_arrays[0]
        )
        self.state = DisplayState(
            initial_time_unit, initial_time_scale, self.mode_switch_threshold
        )

        # Initialize matplotlib figure and axes to None
        self.fig: Optional[mpl.figure.Figure] = None
        self.ax: Optional[mpl.axes.Axes] = None

        # Store visualization elements for each trace
        self._signal_lines: List[mpl.lines.Line2D] = []
        self._envelope_fills: List[Optional[mpl.collections.PolyCollection]] = [
            None
        ] * self.data.num_traces

        # Visualization elements with mode control (definitions, not plot objects)
        self._lines: List[List[Dict[str, Any]]] = [
            [] for _ in range(self.data.num_traces)
        ]
        self._ribbons: List[List[Dict[str, Any]]] = [
            [] for _ in range(self.data.num_traces)
        ]
        self._regions: List[List[Dict[str, Any]]] = [
            [] for _ in range(self.data.num_traces)
        ]
        self._envelopes: List[List[Dict[str, Any]]] = [
            [] for _ in range(self.data.num_traces)
        ]

        # Line objects for each trace (will be populated as needed during rendering)
        self._line_objects: List[List[mpl.artist.Artist]] = [
            [] for _ in range(self.data.num_traces)
        ]  # Changed type hint to Artist
        self._ribbon_objects: List[List[mpl.collections.PolyCollection]] = [
            [] for _ in range(self.data.num_traces)
        ]
        self._region_objects: List[List[mpl.collections.PolyCollection]] = [
            [] for _ in range(self.data.num_traces)
        ]

        # Store current plot data for access by other methods
        self._current_plot_data = {}

        # Initialize coordinate manager
        self.coord_manager = CoordinateManager(self.state)

        # Store initial view for home button (using global time range)
        t_start, t_end = self.data.get_global_time_range()
        self._initial_xlim_raw = (t_start, t_end)

        # Legend state for optimization
        self._current_legend_handles: List[mpl.artist.Artist] = []
        self._current_legend_labels: List[str] = []
        self._legend: Optional[mpl.legend.Legend] = None

        # Track last mode for each trace to optimize element updates
        self._last_mode: Dict[int, Optional[int]] = {
            i: None for i in range(self.data.num_traces)
        }

        # Store original toolbar methods for restoration
        self._original_home = None
        self._original_push_current = None

    def save(self, filepath: str) -> None:
        """
        Save the current plot to a file.

        Parameters
        ----------
        filepath : str
            Path to save the plot image.
        """
        if self.fig is None or self.ax is None:
            raise RuntimeError("Plot has not been initialized yet.")
        self.fig.savefig(filepath)
        logger.info(f"Plot saved to {filepath}")

    def add_line(
        self,
        t: Union[np.ndarray, List[np.ndarray]],
        data: Union[np.ndarray, List[np.ndarray]],
        label: str = "Line",
        color: Optional[str] = None,
        alpha: float = 0.75,
        linestyle: str = "-",
        linewidth: float = 1.0,
        display_mode: int = MODE_BOTH,
        trace_idx: int = 0,
        zorder: int = 5,
    ) -> None:
        """
        Add a line to the plot with mode control.

        Parameters
        ----------
        t : Union[np.ndarray, List[np.ndarray]]
            Time array(s) for the line data. Must match the length of data.
        data : Union[np.ndarray, List[np.ndarray]]
            Line data array(s). Can be a single array or a list of arrays.
        label : str, default="Line"
            Label for the legend.
        color : Optional[str], default=None
            Color for the line. If None, the trace color will be used.
        alpha : float, default=0.75
            Alpha (transparency) for the line.
        linestyle : str, default="-"
            Line style.
        linewidth : float, default=1.0
            Line width.
        display_mode : int, default=MODE_BOTH
            Which mode(s) to show this line in (MODE_ENVELOPE, MODE_DETAIL, or MODE_BOTH).
        trace_idx : int, default=0
            Index of the trace to add the line to.
        zorder : int, default=5
            Z-order for the line (higher values appear on top).
        """
        if trace_idx < 0 or trace_idx >= self.data.num_traces:
            raise ValueError(
                f"Invalid trace index: {trace_idx}. Must be between 0 and {self.data.num_traces - 1}."
            )

        # Validate data length
        if isinstance(data, list):
            if len(data) != len(t):
                raise ValueError(
                    f"Line data length ({len(data)}) must match time array length ({len(t)})."
                )
        else:
            if len(data) != len(t):
                raise ValueError(
                    f"Line data length ({len(data)}) must match time array length ({len(t)})."
                )

        # Use trace color if none provided
        if color is None:
            color = self.data.get_trace_color(trace_idx)

        # Convert inputs to numpy arrays
        t_array = np.asarray(t, dtype=np.float32)
        data_array = np.asarray(data, dtype=np.float32)

        # Assign a unique ID for this custom line for pre-decimation caching
        # We use a negative ID to distinguish from main traces (which use 0, 1, 2...)
        # and ensure uniqueness across custom lines.
        line_id = -(len(self._lines[trace_idx]) + 1)  # Negative, unique per trace

        # Pre-decimate this custom line's data for envelope view
        self.decimator.pre_decimate_data(
            data_id=line_id,
            t=t_array,
            x=data_array,
            max_points=self.max_plot_points,
            envelope_window_samples=None,  # Envelope window calculated automatically
        )

        # Store line definition with raw data and its assigned ID
        line_def = {
            "id": line_id,  # Store the ID for retrieval from decimator
            "t_raw": t_array,  # Store raw time array
            "data_raw": data_array,  # Store raw data array
            "label": label,
            "color": color,
            "alpha": alpha,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "display_mode": display_mode,
            "zorder": zorder,
        }

        logger.debug(
            f"Adding line '{label}' with display_mode={display_mode} (MODE_ENVELOPE={self.MODE_ENVELOPE}, MODE_DETAIL={self.MODE_DETAIL}, MODE_BOTH={self.MODE_BOTH})"
        )
        self._lines[trace_idx].append(line_def)

    def add_ribbon(
        self,
        t: Union[np.ndarray, List[np.ndarray]],
        center_data: Union[np.ndarray, List[np.ndarray]],
        width: Union[float, np.ndarray],
        label: str = "Ribbon",
        color: str = "gray",
        alpha: float = 0.3,
        display_mode: int = MODE_DETAIL,
        trace_idx: int = 0,
        zorder: int = 2,
    ) -> None:
        """
        Add a ribbon (center ± width) with mode control.

        Parameters
        ----------
        t : Union[np.ndarray, List[np.ndarray]]
            Time array(s) for the ribbon data. Must match the length of center_data.
        center_data : Union[np.ndarray, List[np.ndarray]]
            Center line data array(s). Can be a single array or a list of arrays.
        width : Union[float, np.ndarray]
            Width of the ribbon. Can be a single value or an array matching center_data.
        label : str, default="Ribbon"
            Label for the legend.
        color : str, default="gray"
            Color for the ribbon.
        alpha : float, default=0.3
            Alpha (transparency) for the ribbon.
        display_mode : int, default=MODE_DETAIL
            Which mode(s) to show this ribbon in (MODE_ENVELOPE, MODE_DETAIL, or MODE_BOTH).
        trace_idx : int, default=0
            Index of the trace to add the ribbon to.
        """
        if trace_idx < 0 or trace_idx >= self.data.num_traces:
            raise ValueError(
                f"Invalid trace index: {trace_idx}. Must be between 0 and {self.data.num_traces - 1}."
            )

        # Validate data length
        if isinstance(center_data, list):
            if len(center_data) != len(t):
                raise ValueError(
                    f"Ribbon center data length ({len(center_data)}) must match time array length ({len(t)})."
                )
        else:
            if len(center_data) != len(t):
                raise ValueError(
                    f"Ribbon center data length ({len(center_data)}) must match time array length ({len(t)})."
                )

        # Convert center data to numpy array
        center_data = np.asarray(center_data, dtype=np.float32)

        # Handle width as scalar or array
        if isinstance(width, (int, float, np.number)):
            width_array = np.ones_like(center_data) * width
        else:
            if len(width) != len(center_data):
                raise ValueError(
                    f"Ribbon width array length ({len(width)}) must match center data length ({len(center_data)})."
                )
            width_array = np.asarray(width, dtype=np.float32)

        # Assign a unique ID for this custom ribbon
        ribbon_id = -(
            len(self._ribbons[trace_idx]) + 1001
        )  # Negative, unique per trace, offset from lines

        # Pre-decimate this custom ribbon's center data for envelope view
        # We only pre-decimate the center, as width is applied later
        self.decimator.pre_decimate_data(
            data_id=ribbon_id,
            t=np.asarray(t, dtype=np.float32),
            x=center_data,
            max_points=self.max_plot_points,
            envelope_window_samples=None,  # Envelope window calculated automatically
        )

        # Store ribbon definition
        ribbon_def = {
            "id": ribbon_id,
            "t_raw": np.asarray(t, dtype=np.float32),
            "center_data_raw": center_data,
            "width_raw": width_array,
            "label": label,
            "color": color,
            "alpha": alpha,
            "display_mode": display_mode,
            "zorder": zorder,
        }

        self._ribbons[trace_idx].append(ribbon_def)

    def add_envelope(
        self,
        min_data: Union[np.ndarray, List[np.ndarray]],
        max_data: Union[np.ndarray, List[np.ndarray]],
        label: str = "Envelope",
        color: Optional[str] = None,
        alpha: float = 0.4,
        display_mode: int = MODE_ENVELOPE,
        trace_idx: int = 0,
        zorder: int = 1,
    ) -> None:
        """
        Add envelope data with mode control.

        Parameters
        ----------
        min_data : Union[np.ndarray, List[np.ndarray]]
            Minimum envelope data array(s). Can be a single array or a list of arrays.
        max_data : Union[np.ndarray, List[np.ndarray]]
            Maximum envelope data array(s). Can be a single array or a list of arrays.
        label : str, default="Envelope"
            Label for the legend.
        color : Optional[str], default=None
            Color for the envelope. If None, the trace color will be used.
        alpha : float, default=0.4
            Alpha (transparency) for the envelope.
        display_mode : int, default=MODE_ENVELOPE
            Which mode(s) to show this envelope in (MODE_ENVELOPE, MODE_DETAIL, or MODE_BOTH).
        trace_idx : int, default=0
            Index of the trace to add the envelope to.
        """
        if trace_idx < 0 or trace_idx >= self.data.num_traces:
            raise ValueError(
                f"Invalid trace index: {trace_idx}. Must be between 0 and {self.data.num_traces - 1}."
            )

        # Validate data length
        if isinstance(min_data, list):
            if len(min_data) != len(self.data.t_arrays[trace_idx]):
                raise ValueError(
                    f"Envelope min data length ({len(min_data)}) must match time array length ({len(self.data.t_arrays[trace_idx])})."
                )
        else:
            if len(min_data) != len(self.data.t_arrays[trace_idx]):
                raise ValueError(
                    f"Envelope min data length ({len(min_data)}) must match time array length ({len(self.data.t_arrays[trace_idx])})."
                )

        if isinstance(max_data, list):
            if len(max_data) != len(self.data.t_arrays[trace_idx]):
                raise ValueError(
                    f"Envelope max data length ({len(max_data)}) must match time array length ({len(self.data.t_arrays[trace_idx])})."
                )
        else:
            if len(max_data) != len(self.data.t_arrays[trace_idx]):
                raise ValueError(
                    f"Envelope max data length ({len(max_data)}) must match time array length ({len(self.data.t_arrays[trace_idx])})."
                )

        # Use trace color if none provided
        if color is None:
            color = self.data.get_trace_color(trace_idx)

        # Assign a unique ID for this custom envelope
        envelope_id = -(
            len(self._envelopes[trace_idx]) + 2001
        )  # Negative, unique per trace, offset from ribbons

        # Pre-decimate this custom envelope's data for envelope view
        # We'll pre-decimate the average of min/max, and store min/max separately
        t_raw = self.data.t_arrays[trace_idx]
        avg_data = (
            np.asarray(min_data, dtype=np.float32)
            + np.asarray(max_data, dtype=np.float32)
        ) / 2

        self.decimator.pre_decimate_data(
            data_id=envelope_id,
            t=t_raw,
            x=avg_data,  # Pass average for decimation
            max_points=self.max_plot_points,
            envelope_window_samples=None,  # Envelope window calculated automatically
        )

        # Store envelope definition
        envelope_def = {
            "id": envelope_id,
            "t_raw": t_raw,
            "min_data_raw": np.asarray(min_data, dtype=np.float32),
            "max_data_raw": np.asarray(max_data, dtype=np.float32),
            "label": label,
            "color": color,
            "alpha": alpha,
            "display_mode": display_mode,
            "zorder": zorder,
        }

        self._envelopes[trace_idx].append(envelope_def)

    def add_regions(
        self,
        regions: np.ndarray,
        label: str = "Regions",
        color: str = "crimson",
        alpha: float = 0.4,
        display_mode: int = MODE_BOTH,
        trace_idx: int = 0,
        zorder: int = -5,
    ) -> None:
        """
        Add region highlights with mode control.

        Parameters
        ----------
        regions : np.ndarray
            Region data array with shape (N, 2) where each row is [start_time, end_time].
        label : str, default="Regions"
            Label for the legend.
        color : str, default="crimson"
            Color for the regions.
        alpha : float, default=0.4
            Alpha (transparency) for the regions.
        display_mode : int, default=MODE_BOTH
            Which mode(s) to show these regions in (MODE_ENVELOPE, MODE_DETAIL, or MODE_BOTH).
        trace_idx : int, default=0
            Index of the trace to add the regions to.
        """
        if trace_idx < 0 or trace_idx >= self.data.num_traces:
            raise ValueError(
                f"Invalid trace index: {trace_idx}. Must be between 0 and {self.data.num_traces - 1}."
            )

        # Validate regions array
        if regions.ndim != 2 or regions.shape[1] != 2:
            raise ValueError(
                f"Regions array must have shape (N, 2), got {regions.shape}."
            )

        # Store regions definition
        region_def = {
            "regions": np.asarray(regions, dtype=np.float32),
            "label": label,
            "color": color,
            "alpha": alpha,
            "display_mode": display_mode,
            "zorder": zorder,
        }

        logger.debug(
            f"Adding regions '{label}' with {len(regions)} entries, display_mode={display_mode}"
        )
        self._regions[trace_idx].append(region_def)

    def _update_signal_display(
        self,
        trace_idx: int,
        t_display: np.ndarray,
        x_data: np.ndarray,
        envelope_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """
        Update signal display with envelope or raw data for a specific trace.

        Parameters
        ----------
        trace_idx : int
            Index of the trace to update.
        t_display : np.ndarray
            Display time array.
        x_data : np.ndarray
            Signal data array.
        envelope_data : Optional[Tuple[np.ndarray, np.ndarray]], default=None
            Tuple of (min, max) envelope data if in envelope mode.
        """
        logger.debug(f"=== _update_signal_display trace {trace_idx} ===")
        logger.debug(
            f"t_display: len={len(t_display)}, range=[{np.min(t_display) if len(t_display) > 0 else 'empty':.6f}, {np.max(t_display) if len(t_display) > 0 else 'empty':.6f}]"
        )
        logger.debug(
            f"x_data: len={len(x_data)}, range=[{np.min(x_data) if len(x_data) > 0 else 'empty':.6f}, {np.max(x_data) if len(x_data) > 0 else 'empty':.6f}]"
        )
        logger.debug(f"envelope_data: {envelope_data is not None}")

        if envelope_data is not None:
            x_min, x_max = envelope_data
            logger.debug(
                f"envelope x_min: len={len(x_min)}, range=[{np.min(x_min) if len(x_min) > 0 else 'empty':.6f}, {np.max(x_min) if len(x_min) > 0 else 'empty':.6f}]"
            )
            logger.debug(
                f"envelope x_max: len={len(x_max)}, range=[{np.max(x_max) if len(x_max) > 0 else 'empty':.6f}, {np.max(x_max) if len(x_max) > 0 else 'empty':.6f}]"
            )
            self._show_envelope_mode(trace_idx, t_display, envelope_data)
        else:
            logger.debug("Showing detail mode (raw signal)")
            self._show_detail_mode(trace_idx, t_display, x_data)

    def _show_envelope_mode(
        self,
        trace_idx: int,
        t_display: np.ndarray,
        envelope_data: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """
        Show envelope display mode for a specific trace.

        Parameters
        ----------
        trace_idx : int
            Index of the trace to update.
        t_display : np.ndarray
            Display time array.
        envelope_data : Tuple[np.ndarray, np.ndarray]
            Tuple of (min, max) envelope data.
        """
        logger.debug(f"=== _show_envelope_mode trace {trace_idx} ===")
        x_min, x_max = envelope_data
        color = self.data.get_trace_color(trace_idx)
        name = self.data.get_trace_name(trace_idx)

        logger.debug(f"Envelope data: x_min len={len(x_min)}, x_max len={len(x_max)}")
        logger.debug(
            f"t_display range: [{np.min(t_display):.6f}, {np.max(t_display):.6f}]"
        )
        logger.debug(f"y_range: [{np.min(x_min):.6f}, {np.max(x_max):.6f}]")

        # Clean up previous displays
        if self._envelope_fills[trace_idx] is not None:
            logger.debug("Removing previous envelope fill")
            self._envelope_fills[trace_idx].remove()

        logger.debug("Hiding signal line")
        self._signal_lines[trace_idx].set_data([], [])
        self._signal_lines[trace_idx].set_visible(False)

        # Show built-in envelope
        logger.debug(
            f"Creating envelope fill with color={color}, alpha={self.envelope_alpha}"
        )
        self._envelope_fills[trace_idx] = self.ax.fill_between(
            t_display,
            x_min,
            x_max,
            alpha=self.envelope_alpha,
            color=color,
            lw=0,
            label=f"Raw envelope ({name})"
            if self.data.num_traces > 1
            else "Raw envelope",
            zorder=1,  # Keep default envelope at zorder=1
        )

        # Set current mode
        self.state.current_mode = "envelope"
        logger.debug("Set current_mode to 'envelope'")

        # Show any custom elements for this mode
        self._show_custom_elements(trace_idx, t_display, self.MODE_ENVELOPE)

    def _show_detail_mode(
        self, trace_idx: int, t_display: np.ndarray, x_data: np.ndarray
    ) -> None:
        """
        Show detail display mode for a specific trace.

        Parameters
        ----------
        trace_idx : int
            Index of the trace to update.
        t_display : np.ndarray
            Display time array.
        x_data : np.ndarray
            Signal data array.
        """
        logger.debug(f"=== _show_detail_mode trace {trace_idx} ===")
        logger.debug(
            f"t_display: len={len(t_display)}, range=[{np.min(t_display) if len(t_display) > 0 else 'empty':.6f}, {np.max(t_display) if len(t_display) > 0 else 'empty':.6f}]"
        )
        logger.debug(
            f"x_data: len={len(x_data)}, range=[{np.min(x_data) if len(x_data) > 0 else 'empty':.6f}, {np.max(x_data) if len(x_data) > 0 else 'empty':.6f}]"
        )

        # Clean up envelope
        if self._envelope_fills[trace_idx] is not None:
            logger.debug("Removing envelope fill")
            self._envelope_fills[trace_idx].remove()
            self._envelope_fills[trace_idx] = None

        # Update signal line
        line = self._signal_lines[trace_idx]
        logger.debug(
            f"Setting signal line data: linewidth={self.signal_line_width}, alpha={self.signal_alpha}"
        )
        line.set_data(t_display, x_data)
        line.set_linewidth(self.signal_line_width)
        line.set_alpha(self.signal_alpha)
        line.set_visible(True)

        # Set current mode
        self.state.current_mode = "detail"
        logger.debug("Set current_mode to 'detail'")

        # Show any custom elements for this mode
        self._show_custom_elements(trace_idx, t_display, self.MODE_DETAIL)

    def _show_custom_elements(
        self, trace_idx: int, t_display: np.ndarray, current_mode: int
    ) -> None:
        """
        Show custom visualization elements for the current mode.

        Parameters
        ----------
        trace_idx : int
            Index of the trace to update.
        t_display : np.ndarray
            Display time array.
        current_mode : int
            Current display mode (MODE_ENVELOPE or MODE_DETAIL).
        """
        logger.debug(
            f"=== _show_custom_elements trace {trace_idx}, current_mode={current_mode} ==="
        )

        last_mode = self._last_mode.get(trace_idx)
        logger.debug(f"Last mode for trace {trace_idx}: {last_mode}")

        # Always clear and recreate elements when view changes, regardless of mode change
        # This ensures custom lines/ribbons are redrawn correctly with current view data
        logger.debug(
            f"Clearing and recreating elements for trace {trace_idx} (mode: {last_mode} -> {current_mode})"
        )
        self._clear_custom_elements(trace_idx)

        # Get current raw x-limits from the main plot data
        # This is crucial for decimating custom lines to the current view
        current_xlim_raw = self.coord_manager.get_current_view_raw(self.ax)

        # Show lines for current mode
        line_objects = []
        for i, line_def in enumerate(self._lines[trace_idx]):
            logger.debug(
                f"Processing line {i} ('{line_def['label']}'): display_mode={line_def['display_mode']}, current_mode={current_mode}"
            )
            if (
                line_def["display_mode"] & current_mode
            ):  # Bitwise check if mode is enabled
                logger.debug(
                    f"Line {i} ('{line_def['label']}') should be visible in mode {current_mode}"
                )

                # Dynamically decimate the line data for the current view
                # Use the same max_plot_points as the main signal for consistency
                # For custom lines, we want mean decimation if in envelope mode, not min/max envelope
                t_line_raw, line_data, _, _ = self.decimator.decimate_for_view(
                    line_def["t_raw"],
                    line_def["data_raw"],
                    current_xlim_raw,  # Decimate to current view
                    self.max_plot_points,
                    use_envelope=(current_mode == self.MODE_ENVELOPE),
                    data_id=line_def[
                        "id"
                    ],  # Pass the custom line's ID for pre-decimated data lookup
                    envelope_window_samples=None,  # Envelope window calculated automatically
                    mode_switch_threshold=self.mode_switch_threshold,  # Pass mode switch threshold
                    return_envelope_min_max=False,  # Custom lines never return min/max envelope
                )

                if len(t_line_raw) == 0 or len(line_data) == 0:
                    logger.warning(
                        f"Line {i} ('{line_def['label']}') has empty data after decimation for current view, skipping plot."
                    )
                    continue

                # Make sure the time array is in display coordinates
                t_line_display = self.coord_manager.raw_to_display(t_line_raw)

                # Always plot as a regular line
                (line,) = self.ax.plot(
                    t_line_display,
                    line_data,
                    label=line_def["label"],
                    color=line_def["color"],
                    alpha=line_def["alpha"],
                    linestyle=line_def["linestyle"],
                    linewidth=line_def["linewidth"],
                    zorder=line_def["zorder"],
                )
                line_objects.append(
                    (line, line_def)
                )  # Store both the line and its definition
                logger.debug(f"Added line {i} ('{line_def['label']}') to plot")
            else:
                logger.debug(
                    f"Line {i} ('{line_def['label']}') should NOT be visible in mode {current_mode}"
                )

        # Show ribbons for current mode
        ribbon_objects = []
        for ribbon_def in self._ribbons[trace_idx]:
            logger.debug(
                f"Processing ribbon ('{ribbon_def['label']}'): display_mode={ribbon_def['display_mode']}, current_mode={current_mode}"
            )
            if ribbon_def["display_mode"] & current_mode:
                logger.debug(
                    f"Ribbon ('{ribbon_def['label']}') should be visible in mode {current_mode}"
                )

                # Ribbons are always plotted as fills, so we need to decimate their center and width
                # We'll treat the center_data as the 'signal' for decimation purposes
                (
                    t_ribbon_raw,
                    center_data_decimated,
                    min_center_envelope,
                    max_center_envelope,
                ) = self.decimator.decimate_for_view(
                    ribbon_def["t_raw"],
                    ribbon_def["center_data_raw"],
                    current_xlim_raw,
                    self.max_plot_points,
                    use_envelope=(
                        current_mode == self.MODE_ENVELOPE
                    ),  # Use envelope for ribbons if in envelope mode
                    data_id=ribbon_def[
                        "id"
                    ],  # Pass the custom ribbon's ID for pre-decimated data lookup
                    return_envelope_min_max=True,  # Ribbons always need min/max to draw fill
                    envelope_window_samples=None,  # Envelope window calculated automatically
                    mode_switch_threshold=self.mode_switch_threshold,
                )

                # Decimate the width array as well, if it's an array
                width_decimated = ribbon_def["width_raw"]
                if len(ribbon_def["width_raw"]) > len(
                    t_ribbon_raw
                ):  # If raw width is longer than decimated time
                    # For simplicity, we'll just take the mean of the width in each bin
                    # A more robust solution might involve passing width as another data stream to decimate_for_view
                    # For now, we'll manually decimate it based on the t_ribbon_raw indices
                    # Find indices in raw data corresponding to decimated time points
                    # This is a simplified approach and assumes uniform sampling for width
                    indices = np.searchsorted(ribbon_def["t_raw"], t_ribbon_raw)
                    indices = np.clip(indices, 0, len(ribbon_def["width_raw"]) - 1)
                    width_decimated = ribbon_def["width_raw"][indices]

                # If the ribbon was decimated to an envelope, use that for min/max
                if (
                    current_mode == self.MODE_ENVELOPE
                    and min_center_envelope is not None
                    and max_center_envelope is not None
                ):
                    lower_bound = min_center_envelope - width_decimated
                    upper_bound = max_center_envelope + width_decimated
                else:
                    lower_bound = center_data_decimated - width_decimated
                    upper_bound = center_data_decimated + width_decimated

                if len(t_ribbon_raw) == 0 or len(lower_bound) == 0:
                    logger.warning(
                        f"Ribbon ('{ribbon_def['label']}') has empty data after decimation, skipping plot."
                    )
                    continue

                # Make sure the time array is in display coordinates
                t_ribbon_display = self.coord_manager.raw_to_display(t_ribbon_raw)

                ribbon = self.ax.fill_between(
                    t_ribbon_display,
                    lower_bound,
                    upper_bound,
                    color=ribbon_def["color"],
                    alpha=ribbon_def["alpha"],
                    label=ribbon_def["label"],
                    zorder=ribbon_def["zorder"],
                )
                ribbon_objects.append(
                    (ribbon, ribbon_def)
                )  # Store both the ribbon and its definition
                logger.debug(f"Added ribbon ('{ribbon_def['label']}') to plot")
            else:
                logger.debug(
                    f"Ribbon ('{ribbon_def['label']}') should NOT be visible in mode {current_mode}"
                )

        # Show custom envelopes for current mode
        for envelope_def in self._envelopes[trace_idx]:
            logger.debug(
                f"Processing custom envelope ('{envelope_def['label']}'): display_mode={envelope_def['display_mode']}, current_mode={current_mode}"
            )
            if envelope_def["display_mode"] & current_mode:
                logger.debug(
                    f"Custom envelope ('{envelope_def['label']}') should be visible in mode {current_mode}"
                )

                # For custom envelopes, we need to handle min/max data specially
                # We'll decimate the min and max data separately using the envelope's stored data
                # Since we stored min/max in the pre-decimated data, we can retrieve them

                # Get the pre-decimated envelope data for this custom envelope
                if envelope_def["id"] in self.decimator._pre_decimated_envelopes:
                    pre_dec_data = self.decimator._pre_decimated_envelopes[
                        envelope_def["id"]
                    ]
                    # The min/max data was stored in bg_initial/bg_clean during pre-decimation
                    t_envelope_raw, _, min_data_decimated, max_data_decimated = (
                        self.decimator.decimate_for_view(
                            envelope_def["t_raw"],
                            (
                                envelope_def["min_data_raw"]
                                + envelope_def["max_data_raw"]
                            )
                            / 2,  # Average for decimation
                            current_xlim_raw,
                            self.max_plot_points,
                            use_envelope=True,  # Always treat custom envelopes as envelopes
                            data_id=envelope_def[
                                "id"
                            ],  # Pass the custom envelope's ID for pre-decimated data lookup
                            return_envelope_min_max=True,  # Custom envelopes always need min/max to draw fill
                            envelope_window_samples=None,  # Envelope window calculated automatically
                            mode_switch_threshold=self.mode_switch_threshold,
                        )
                    )
                    # For custom envelopes, the min/max are returned directly as the last two return values
                else:
                    # Fallback if no pre-decimated data
                    logger.warning(
                        f"No pre-decimated data for custom envelope {envelope_def['id']}, using raw decimation"
                    )
                    t_envelope_raw, _, min_data_decimated, max_data_decimated = (
                        self.decimator.decimate_for_view(
                            envelope_def["t_raw"],
                            (
                                envelope_def["min_data_raw"]
                                + envelope_def["max_data_raw"]
                            )
                            / 2,
                            current_xlim_raw,
                            self.max_plot_points,
                            use_envelope=True,
                            data_id=None,  # No pre-decimated data available
                            return_envelope_min_max=True,
                            envelope_window_samples=None,  # Envelope window calculated automatically
                            mode_switch_threshold=self.mode_switch_threshold,
                        )
                    )

                if (
                    len(t_envelope_raw) == 0
                    or min_data_decimated is None
                    or max_data_decimated is None
                    or len(min_data_decimated) == 0
                ):
                    logger.warning(
                        f"Custom envelope ('{envelope_def['label']}') has empty data after decimation, skipping plot."
                    )
                    continue

                t_envelope_display = self.coord_manager.raw_to_display(t_envelope_raw)

                envelope = self.ax.fill_between(
                    t_envelope_display,
                    min_data_decimated,
                    max_data_decimated,
                    color=envelope_def["color"],
                    alpha=envelope_def["alpha"],
                    label=envelope_def["label"],
                    zorder=envelope_def["zorder"],
                )
                ribbon_objects.append(
                    (envelope, envelope_def)
                )  # Store in ribbon objects
                logger.debug(
                    f"Added custom envelope ('{envelope_def['label']}') to plot"
                )
            else:
                logger.debug(
                    f"Custom envelope ('{envelope_def['label']}') should NOT be visible in mode {current_mode}"
                )

        # Store objects with their definitions for future updates
        self._line_objects[trace_idx] = line_objects
        self._ribbon_objects[trace_idx] = ribbon_objects

        # Update last mode AFTER processing
        self._last_mode[trace_idx] = current_mode

    def _update_element_visibility(self, trace_idx: int, current_mode: int) -> None:
        """
        Update visibility of existing custom elements based on current mode.

        Parameters
        ----------
        trace_idx : int
            Index of the trace to update.
        current_mode : int
            Current display mode (MODE_ENVELOPE or MODE_DETAIL).
        """
        logger.debug(
            f"Updating element visibility for trace {trace_idx}, current_mode={current_mode}"
        )
        # Update line visibility
        for line_obj, line_def in self._line_objects[trace_idx]:
            should_be_visible = bool(line_def["display_mode"] & current_mode)
            if line_obj.get_visible() != should_be_visible:
                line_obj.set_visible(should_be_visible)
                logger.debug(
                    f"Set visibility of line '{line_def['label']}' to {should_be_visible}"
                )

        # Update ribbon visibility
        for ribbon_obj, ribbon_def in self._ribbon_objects[trace_idx]:
            should_be_visible = bool(ribbon_def["display_mode"] & current_mode)
            if ribbon_obj.get_visible() != should_be_visible:
                ribbon_obj.set_visible(should_be_visible)
                logger.debug(
                    f"Set visibility of ribbon '{ribbon_def['label']}' to {should_be_visible}"
                )

    def _clear_custom_elements(self, trace_idx: int) -> None:
        """
        Clear all custom visualization elements for a trace.

        Parameters
        ----------
        trace_idx : int
            Index of the trace to clear elements for.
        """
        logger.debug(f"Clearing custom elements for trace {trace_idx}")
        # Clear lines
        for line_obj, _ in self._line_objects[trace_idx]:
            line_obj.remove()
        self._line_objects[trace_idx].clear()

        # Clear ribbons
        for ribbon_obj, _ in self._ribbon_objects[trace_idx]:
            ribbon_obj.remove()
        self._ribbon_objects[trace_idx].clear()

    def _update_tick_locator(self, time_span_raw: np.float32) -> None:
        """Update tick locator based on current time scale and span."""
        if self.state.current_time_scale >= np.float32(1e6):  # microseconds or smaller
            # For microsecond scale, use reasonable intervals
            tick_interval = max(
                1, int(time_span_raw * self.state.current_time_scale / 10)
            )
            self.ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
        else:
            # For larger scales, use matplotlib's default auto locator
            self.ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())

    def _update_legend(self) -> None:
        """Updates the plot legend, filtering out invisible elements and optimising rebuilds."""
        logger.debug("Updating legend...")
        handles, labels = self.ax.get_legend_handles_labels()

        # Filter for unique and visible handles/labels
        unique_labels = []
        unique_handles = []
        for h, l in zip(handles, labels):
            # Check if the handle has a get_visible method and if it returns True
            # For fill_between objects (ribbons, envelopes, regions), get_visible might not exist or behave differently
            # For these, we assume they are visible if they are in the list of objects
            is_visible = True
            if hasattr(h, "get_visible"):
                is_visible = h.get_visible()
            elif isinstance(
                h, mpl.collections.PolyCollection
            ):  # For fill_between objects
                # PolyCollection doesn't have get_visible, but its patches might.
                # Or we can assume it's visible if it's part of the current plot.
                # For now, assume it's visible if it's a PolyCollection and has data.
                is_visible = len(h.get_paths()) > 0  # Check if it has any paths to draw

            if l not in unique_labels and is_visible:
                unique_labels.append(l)
                unique_handles.append(h)

        logger.debug(f"Unique visible legend items found: {unique_labels}")

        # Create a hash of current handles/labels for efficient comparison
        current_hash = hash(tuple(id(h) for h in unique_handles) + tuple(unique_labels))

        # Check if legend content actually changed
        if (
            not hasattr(self, "_last_legend_hash")
            or self._last_legend_hash != current_hash
        ):
            logger.debug("Legend content changed, rebuilding legend.")
            if self._legend is not None:
                self._legend.remove()  # Remove old legend to prevent duplicates

            if unique_handles:  # Only create legend if there are handles to show
                self._legend = self.ax.legend(
                    unique_handles, unique_labels, loc="lower right"
                )
                logger.debug("New legend created.")
            else:
                self._legend = None  # No legend to show
                logger.debug("No legend to show.")

            self._current_legend_handles = unique_handles
            self._current_legend_labels = unique_labels
            self._last_legend_hash = current_hash
        else:
            logger.debug("Legend content unchanged, skipping rebuild.")

    def _clear_navigation_history(self):
        """Clear matplotlib's navigation history when coordinate system changes."""
        if (
            self.fig
            and self.fig.canvas
            and hasattr(self.fig.canvas, "toolbar")
            and self.fig.canvas.toolbar
        ):
            toolbar = self.fig.canvas.toolbar
            if hasattr(toolbar, "_nav_stack"):
                toolbar._nav_stack.clear()

    def _push_current_view(self):
        """Push current view to navigation history as new base."""
        if (
            self.fig
            and self.fig.canvas
            and hasattr(self.fig.canvas, "toolbar")
            and self.fig.canvas.toolbar
        ):
            toolbar = self.fig.canvas.toolbar
            if hasattr(toolbar, "push_current"):
                toolbar.push_current()

    def _update_axis_formatting(self) -> None:
        """Update axis labels and formatters."""
        if self.state.offset_time_raw is not None:
            offset_value = self.state.offset_time_raw * (
                1e3
                if self.state.offset_unit == "ms"
                else 1e6
                if self.state.offset_unit == "us"
                else 1e9
                if self.state.offset_unit == "ns"
                else 1.0
            )
            xlabel = f"Time ({self.state.current_time_unit}) + {offset_value:.3g} {self.state.offset_unit}"
        else:
            xlabel = f"Time ({self.state.current_time_unit})"

        self.ax.set_xlabel(xlabel)

        formatter = _create_time_formatter(
            self.state.offset_time_raw, self.state.current_time_scale
        )
        self.ax.xaxis.set_major_formatter(formatter)

    def _update_overlay_lines(
        self, plot_data: Dict[str, Any], show_overlays: bool
    ) -> None:
        """Update overlay lines based on zoom level and data availability."""
        # Clear existing overlay lines from the plot
        # This method is not currently used in the provided code, but if it were,
        # it would need to be updated to use the new decimation strategy.
        # For now, leaving it as is, assuming it's a placeholder or for future_use.
        # If it were to be used, it would need to call decimate_for_view for each overlay line.
        pass  # No _overlay_lines attribute in this class, this method is unused.

    def _update_y_limits(self, plot_data: Dict[str, Any], use_envelope: bool) -> None:
        """Update y-axis limits to fit current data."""
        y_min_data = float("inf")
        y_max_data = float("-inf")

        # Process each trace
        for trace_idx in range(self.data.num_traces):
            x_new_key = f"x_new_{trace_idx}"
            x_min_key = f"x_min_{trace_idx}"
            x_max_key = f"x_max_{trace_idx}"

            if x_new_key not in plot_data:
                continue

            # Include signal data
            if len(plot_data[x_new_key]) > 0:
                y_min_data = min(y_min_data, np.min(plot_data[x_new_key]))
                y_max_data = max(y_max_data, np.max(plot_data[x_new_key]))

            # Include envelope data if available
            if use_envelope and x_min_key in plot_data and x_max_key in plot_data:
                if (
                    plot_data[x_min_key] is not None
                    and plot_data[x_max_key] is not None
                    and len(plot_data[x_min_key]) > 0
                ):
                    y_min_data = min(y_min_data, np.min(plot_data[x_min_key]))
                    y_max_data = max(y_max_data, np.max(plot_data[x_max_key]))

            # Include custom lines
            for line_obj, _ in self._line_objects[trace_idx]:
                # Check if line_obj is a Line2D or PolyCollection
                if isinstance(line_obj, mpl.lines.Line2D):
                    y_data = line_obj.get_ydata()
                    if len(y_data) > 0:
                        y_min_data = min(y_min_data, np.min(y_data))
                        y_max_data = max(y_max_data, np.max(y_data))
                elif isinstance(line_obj, mpl.collections.PolyCollection):
                    # For fill_between objects, iterate through paths to get y-coordinates
                    for path in line_obj.get_paths():
                        vertices = path.vertices
                        if len(vertices) > 0:
                            y_min_data = min(y_min_data, np.min(vertices[:, 1]))
                            y_max_data = max(y_max_data, np.max(vertices[:, 1]))

            # Include ribbon data
            for ribbon_obj, _ in self._ribbon_objects[trace_idx]:
                # For fill_between objects, we need to get the paths
                if hasattr(ribbon_obj, "get_paths") and len(ribbon_obj.get_paths()) > 0:
                    for path in ribbon_obj.get_paths():
                        vertices = path.vertices
                        if len(vertices) > 0:
                            y_min_data = min(y_min_data, np.min(vertices[:, 1]))
                            y_max_data = max(y_max_data, np.max(vertices[:, 1]))

        # Handle case where no data was found
        if y_min_data == float("inf") or y_max_data == float("-inf"):
            self.ax.set_ylim(0, 1)
            return

        data_range = y_max_data - y_min_data
        data_mean = (y_min_data + y_max_data) / 2

        # Use min_y_range to ensure a minimum visible range
        min_visible_range = self.min_y_range

        if data_range < min_visible_range:
            y_min = data_mean - min_visible_range / 2
            y_max = data_mean + min_visible_range / 2
        else:
            y_margin = self.y_margin_fraction * data_range
            y_min = y_min_data - y_margin
            y_max = y_max_data + y_margin

        logger.debug(
            f"Y-limit calculation details: data_range={data_range:.3g}, min_visible_range={min_visible_range:.3g}, data_mean={data_mean:.3g}"
        )  # ADDED THIS LINE
        logger.debug(
            f"Pre-set Y-limits: y_min={y_min:.9f}, y_max={y_max:.9f}"
        )  # ADDED THIS LINE
        self.ax.set_ylim(y_min, y_max)

    def _update_plot_data(self, ax_obj) -> None:
        """Update plot based on current view."""
        if self.state.is_updating():
            return

        self.state.set_updating(True)

        try:
            try:
                # Add debug logging for current axis limits
                display_xlim = ax_obj.get_xlim()
                logger.debug(f"Current display xlim: {display_xlim}")

                view_params = self._calculate_view_parameters(ax_obj)
                logger.debug(
                    f"Calculated view parameters: xlim_raw={view_params['xlim_raw']}, time_span_raw={view_params['time_span_raw']}, use_envelope={view_params['use_envelope']}"
                )

                plot_data = self._get_plot_data(view_params)

                # Debug data availability
                data_summary = {}
                for trace_idx in range(self.data.num_traces):
                    t_key = f"t_display_{trace_idx}"
                    if t_key in plot_data:
                        data_summary[t_key] = len(plot_data[t_key])
                logger.debug(f"Plot data summary: {data_summary}")

                self._render_plot_elements(plot_data, view_params)
                self._update_regions_and_legend(view_params["xlim_display"])
                self.fig.canvas.draw_idle()
            except Exception as e:
                logger.exception(f"Error updating plot: {e}")
                # Try to recover by resetting to home view
                logger.info("Attempting to recover by resetting to home view")
                self.home()
        finally:
            self.state.set_updating(False)

    def _calculate_view_parameters(self, ax_obj) -> Dict[str, Any]:
        """Calculate view parameters from current axis state."""
        try:
            xlim_raw = self.coord_manager.get_current_view_raw(ax_obj)

            # Validate xlim_raw values
            if not np.isfinite(xlim_raw[0]) or not np.isfinite(xlim_raw[1]):
                logger.warning(
                    f"Invalid xlim_raw from axis: {xlim_raw}. Using initial view."
                )
                xlim_raw = self._initial_xlim_raw

            # Ensure xlim_raw is in ascending order
            if xlim_raw[0] > xlim_raw[1]:
                logger.warning(f"xlim_raw values out of order: {xlim_raw}. Swapping.")
                xlim_raw = (xlim_raw[1], xlim_raw[0])

            time_span_raw = xlim_raw[1] - xlim_raw[0]
            use_envelope = self.state.should_use_envelope(time_span_raw)
            current_mode = self.MODE_ENVELOPE if use_envelope else self.MODE_DETAIL

            logger.debug(f"=== _calculate_view_parameters ===")
            logger.debug(f"xlim_raw: {xlim_raw}")
            logger.debug(f"time_span_raw: {time_span_raw:.6e}s")
            logger.debug(
                f"envelope_limit: {self.mode_switch_threshold:.6e}s"
            )  # Use mode_switch_threshold
            logger.debug(f"use_envelope: {use_envelope}")
            logger.debug(
                f"current_mode: {current_mode} ({'ENVELOPE' if current_mode == self.MODE_ENVELOPE else 'DETAIL'})"
            )

            # Update coordinate system if needed
            coordinate_system_changed = self.state.update_display_params(
                xlim_raw, time_span_raw
            )
            if coordinate_system_changed:
                logger.debug("Coordinate system changed, updating")
                self._update_coordinate_system(xlim_raw, time_span_raw)

            return {
                "xlim_raw": xlim_raw,
                "time_span_raw": time_span_raw,
                "xlim_display": self.coord_manager.xlim_raw_to_display(xlim_raw),
                "use_envelope": use_envelope,
                "current_mode": current_mode,
            }
        except Exception as e:
            logger.exception(f"Error calculating view parameters: {e}")
            # Return safe default values
            return {
                "xlim_raw": self._initial_xlim_raw,
                "time_span_raw": self._initial_xlim_raw[1] - self._initial_xlim_raw[0],
                "xlim_display": self.coord_manager.xlim_raw_to_display(
                    self._initial_xlim_raw
                ),
                "use_envelope": True,
                "current_mode": self.MODE_ENVELOPE,
            }

    def _get_plot_data(self, view_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get decimated plot data for current view."""
        logger.debug(f"=== _get_plot_data ===")
        logger.debug(f"view_params: {view_params}")

        plot_data = {}

        # Process each trace
        for trace_idx in range(self.data.num_traces):
            logger.debug(f"--- Processing trace {trace_idx} ---")
            t_arr = self.data.t_arrays[trace_idx]
            x_arr = self.data.x_arrays[trace_idx]

            logger.debug(f"Input data: t_arr len={len(t_arr)}, x_arr len={len(x_arr)}")

            try:
                t_raw, x_new, x_min, x_max = self.decimator.decimate_for_view(
                    t_arr,
                    x_arr,
                    view_params["xlim_raw"],
                    self.max_plot_points,
                    view_params["use_envelope"],
                    trace_idx,  # Pass trace_id to use pre-decimated data
                    envelope_window_samples=None,  # Envelope window calculated automatically
                    mode_switch_threshold=self.mode_switch_threshold,  # Pass mode switch threshold
                    return_envelope_min_max=True,  # Main signal always returns envelope min/max if use_envelope is True
                )

                logger.debug(
                    f"Decimated data: t_raw len={len(t_raw)}, x_new len={len(x_new)}"
                )
                logger.debug(
                    f"Envelope data: x_min={'None' if x_min is None else f'len={len(x_min)}'}, x_max={'None' if x_max is None else f'len={len(x_max)}'}"
                )

                if len(t_raw) == 0:
                    logger.warning(
                        f"No data in current view for trace {trace_idx}. View range: {view_params['xlim_raw']}"
                    )
                    # Add empty arrays for this trace
                    plot_data[f"t_display_{trace_idx}"] = np.array([], dtype=np.float32)
                    plot_data[f"x_new_{trace_idx}"] = np.array([], dtype=np.float32)
                    plot_data[f"x_min_{trace_idx}"] = None
                    plot_data[f"x_max_{trace_idx}"] = None
                    continue

                t_display = self.coord_manager.raw_to_display(t_raw)
                logger.debug(
                    f"Converted to display coordinates: t_display range=[{np.min(t_display):.6f}, {np.max(t_display):.6f}]"
                )

                # Store data for this trace
                plot_data[f"t_display_{trace_idx}"] = t_display
                plot_data[f"x_new_{trace_idx}"] = x_new
                plot_data[f"x_min_{trace_idx}"] = x_min
                plot_data[f"x_max_{trace_idx}"] = x_max

                logger.debug(f"Stored plot data for trace {trace_idx}")
            except Exception as e:
                logger.exception(f"Error getting plot data for trace {trace_idx}: {e}")
                # Add empty arrays for this trace to prevent further errors
                plot_data[f"t_display_{trace_idx}"] = np.array([], dtype=np.float32)
                plot_data[f"x_new_{trace_idx}"] = np.array([], dtype=np.float32)
                plot_data[f"x_min_{trace_idx}"] = None
                plot_data[f"x_max_{trace_idx}"] = None

        logger.debug(f"Final plot_data keys: {list(plot_data.keys())}")
        return plot_data

    def _render_plot_elements(
        self, plot_data: Dict[str, Any], view_params: Dict[str, Any]
    ) -> None:
        """Render all plot elements with current data."""
        logger.debug(f"=== _render_plot_elements ===")
        logger.debug(f"view_params use_envelope: {view_params['use_envelope']}")

        # Store the current plot data for use by other methods
        self._current_plot_data = plot_data

        # Check if we have any data to plot
        has_data = False
        data_summary = {}
        for trace_idx in range(self.data.num_traces):
            key = f"t_display_{trace_idx}"
            if key in plot_data and len(plot_data[key]) > 0:
                has_data = True
                data_summary[f"trace_{trace_idx}"] = len(plot_data[key])
            else:
                data_summary[f"trace_{trace_idx}"] = 0

        logger.debug(f"Data summary: {data_summary}, has_data: {has_data}")

        if not has_data:
            logger.warning("No data to plot, clearing all elements")
            # If no data, clear all lines and return
            for i in range(self.data.num_traces):
                self._signal_lines[i].set_data([], [])
                if self._envelope_fills[i] is not None:
                    self._envelope_fills[i].remove()
                    self._envelope_fills[i] = None

                # Clear custom elements
                self._clear_custom_elements(i)

            self.ax.set_ylim(0, 1)  # Set a default y-limit
            return

        # Process each trace
        for trace_idx in range(self.data.num_traces):
            logger.debug(f"--- Rendering trace {trace_idx} ---")
            t_display_key = f"t_display_{trace_idx}"
            x_new_key = f"x_new_{trace_idx}"
            x_min_key = f"x_min_{trace_idx}"
            x_max_key = f"x_max_{trace_idx}"

            if t_display_key not in plot_data or len(plot_data[t_display_key]) == 0:
                logger.debug(f"No data for trace {trace_idx}, hiding elements")
                # No data for this trace, hide its elements
                self._signal_lines[trace_idx].set_data([], [])
                if self._envelope_fills[trace_idx] is not None:
                    self._envelope_fills[trace_idx].remove()
                    self._envelope_fills[trace_idx] = None

                # Clear custom elements
                self._clear_custom_elements(trace_idx)
                continue

            # Update signal display
            envelope_data = None
            if (
                view_params["use_envelope"]
                and x_min_key in plot_data
                and x_max_key in plot_data
            ):
                if (
                    plot_data[x_min_key] is not None
                    and plot_data[x_max_key] is not None
                ):
                    envelope_data = (plot_data[x_min_key], plot_data[x_max_key])
                    logger.debug(f"Using envelope data for trace {trace_idx}")
                else:
                    logger.debug(
                        f"Envelope mode requested but no envelope data for trace {trace_idx}"
                    )
            else:
                logger.debug(f"Detail mode for trace {trace_idx}")

            self._update_signal_display(
                trace_idx, plot_data[t_display_key], plot_data[x_new_key], envelope_data
            )

        # Update y-limits
        logger.debug("Updating y-limits")
        self._update_y_limits(plot_data, view_params["use_envelope"])

    def _update_coordinate_system(
        self, xlim_raw: Tuple[np.float32, np.float32], time_span_raw: np.float32
    ) -> None:
        """Update coordinate system and axis formatting."""
        self._clear_region_fills()
        self._update_axis_formatting()
        self._update_tick_locator(time_span_raw)

        xlim_display = self.coord_manager.xlim_raw_to_display(xlim_raw)
        self.ax.set_xlim(xlim_display)

        self._clear_navigation_history()
        self._push_current_view()

    def _update_regions_and_legend(
        self, xlim_display: Tuple[np.float32, np.float32]
    ) -> None:
        """Update regions and legend."""
        self._refresh_region_display(xlim_display)
        self._update_legend()

    def _refresh_region_display(
        self, xlim_display: Tuple[np.float32, np.float32]
    ) -> None:
        """Refresh region display for current view."""
        logger.debug(f"=== _refresh_region_display ===")
        self._clear_region_fills()

        # Get current mode
        current_mode = (
            self.MODE_ENVELOPE
            if self.state.current_mode == "envelope"
            else self.MODE_DETAIL
        )
        logger.debug(f"Current display mode for regions: {current_mode}")

        for trace_idx in range(self.data.num_traces):
            logger.debug(f"Processing regions for trace {trace_idx}")
            # Process each region definition
            for region_def in self._regions[trace_idx]:
                logger.debug(
                    f"Region '{region_def['label']}': display_mode={region_def['display_mode']}, current_mode={current_mode}"
                )
                # Skip if not visible in current mode
                if not (region_def["display_mode"] & current_mode):
                    logger.debug(
                        f"Region '{region_def['label']}' not visible in current mode {current_mode}, skipping."
                    )
                    continue

                regions = region_def["regions"]
                if regions is None or len(regions) == 0:
                    logger.debug(
                        f"No regions data for '{region_def['label']}', skipping."
                    )
                    continue

                logger.debug(
                    f"Displaying {len(regions)} regions for '{region_def['label']}' in mode {current_mode}"
                )

                color = region_def["color"]
                label = region_def["label"]
                alpha = region_def["alpha"]
                first_visible_region = True

                for t_start, t_end in regions:
                    t_start_display = self.coord_manager.raw_to_display(t_start)
                    t_end_display = self.coord_manager.raw_to_display(t_end)

                    # Check if region overlaps with current view
                    if not (
                        t_end_display <= xlim_display[0]
                        or t_start_display >= xlim_display[1]
                    ):
                        # Only show label for first visible region
                        current_label = label if first_visible_region else ""
                        if first_visible_region and len(regions) > 1:
                            current_label = f"{label} ({len(regions)})"

                        logger.debug(
                            f"Adding region span from {t_start_display:.6f} to {t_end_display:.6f} (raw: {t_start:.6f} to {t_end:.6f}) for '{label}'"
                        )
                        fill = self.ax.axvspan(
                            t_start_display,
                            t_end_display,
                            alpha=alpha,
                            color=color,
                            linewidth=0.5,
                            label=current_label,
                            zorder=region_def["zorder"],
                        )
                        self._region_objects[trace_idx].append((fill, region_def))
                        first_visible_region = False
                    else:
                        logger.debug(
                            f"Region span from {t_start_display:.6f} to {t_end_display:.6f} (raw: {t_start:.6f} to {t_end:.6f}) for '{label}' is outside current view {xlim_display}, skipping."
                        )

    def _clear_region_fills(self) -> None:
        """Clear all region fills."""
        logger.debug("Clearing region fills.")
        for trace_fills in self._region_objects:
            for fill_item in trace_fills:
                # Handle both old format (just fill object) and new format (tuple)
                if isinstance(fill_item, tuple):
                    fill, _ = fill_item  # Extract the fill object from the tuple
                    fill.remove()
                else:
                    fill_item.remove()  # Old format - direct fill object
            trace_fills.clear()
        logger.debug("Region fills cleared.")

    def _setup_plot_elements(self) -> None:
        """
        Initialise matplotlib plot elements (lines, fills) for each trace.
        This is called once during render().
        """
        if self.fig is None or self.ax is None:
            raise RuntimeError(
                "Figure and Axes must be created before setting up plot elements."
            )

        # Create initial signal line objects for each trace
        for i in range(self.data.num_traces):
            color = self.data.get_trace_color(i)
            name = self.data.get_trace_name(i)

            # Signal line
            (line_signal,) = self.ax.plot(
                [],
                [],
                label="Raw data" if self.data.num_traces == 1 else f"Raw data ({name})",
                color=color,
                alpha=self.signal_alpha,
            )
            self._signal_lines.append(line_signal)

    def _connect_callbacks(self) -> None:
        """Connect matplotlib callbacks."""
        if self.ax is None:
            raise RuntimeError("Axes must be created before connecting callbacks.")
        self.ax.callbacks.connect("xlim_changed", self._update_plot_data)

    def _setup_toolbar_overrides(self) -> None:
        """Override matplotlib toolbar methods (e.g., home button)."""
        if (
            self.fig
            and self.fig.canvas
            and hasattr(self.fig.canvas, "toolbar")
            and self.fig.canvas.toolbar
        ):
            toolbar = self.fig.canvas.toolbar

            # Store original methods
            self._original_home = getattr(toolbar, "home", None)
            self._original_push_current = getattr(toolbar, "push_current", None)

            # Create our custom home method
            def custom_home(*args, **kwargs):
                logger.debug("Toolbar home button pressed - calling custom home")
                self.home()

            # Override both the method and try to find the actual button
            toolbar.home = custom_home

            # For Qt backend, also override the action
            if hasattr(toolbar, "actions"):
                for action in toolbar.actions():
                    if hasattr(action, "text") and hasattr(action, "objectName"):
                        action_text = (
                            action.text() if callable(action.text) else str(action.text)
                        )
                        action_name = (
                            action.objectName()
                            if callable(action.objectName)
                            else str(action.objectName)
                        )
                        if action_text == "Home" or "home" in action_name.lower():
                            if hasattr(action, "triggered"):
                                action.triggered.disconnect()
                                action.triggered.connect(custom_home)
                                logger.debug("Connected custom home to Qt action")
                                break

            # For other backends, try to override the button callback
            if hasattr(toolbar, "_buttons") and "Home" in toolbar._buttons:
                home_button = toolbar._buttons["Home"]
                if hasattr(home_button, "configure"):
                    home_button.configure(command=custom_home)
                    logger.debug("Connected custom home to Tkinter button")

    def _set_initial_view_and_labels(self) -> None:
        """Set initial axis limits, title, and labels."""
        if self.ax is None:
            raise RuntimeError(
                "Axes must be created before setting initial view and labels."
            )

        # Create title based on number of traces
        if self.data.num_traces == 1:
            self.ax.set_title(f"{self.data.names[0]}")
        else:
            # Multiple traces - just show "Multiple Traces"
            self.ax.set_title(f"Multiple Traces ({self.data.num_traces})")
        self.ax.set_xlabel(f"Time ({self.state.current_time_unit})")
        self.ax.set_ylabel("Signal")

        # Set initial xlim
        initial_xlim_display = self.coord_manager.xlim_raw_to_display(
            self._initial_xlim_raw
        )
        self.ax.set_xlim(initial_xlim_display)

    def render(self) -> None:
        """
        Renders the oscilloscope plot. This method must be called after all
        data and visualization elements have been added.
        """
        if self.fig is not None or self.ax is not None:
            logger.warning(
                "Plot already rendered. Call `home()` to reset or create a new instance."
            )
            return

        logger.info("Rendering plot...")
        self.fig, self.ax = plt.subplots(figsize=(10, 5))

        self._setup_plot_elements()
        self._connect_callbacks()
        self._setup_toolbar_overrides()
        self._set_initial_view_and_labels()

        # Calculate initial parameters for the full view
        t_start, t_end = self.data.get_global_time_range()
        full_time_span = t_end - t_start

        logger.info(
            f"Initial render: full time span={full_time_span:.3e}s, envelope_limit={self.mode_switch_threshold:.3e}s"
        )

        # Set initial display state based on full view
        self.state.current_time_unit, self.state.current_time_scale = (
            _get_optimal_time_unit_and_scale(full_time_span)
        )
        self.state.current_mode = (
            "envelope" if self.state.should_use_envelope(full_time_span) else "detail"
        )

        # Force initial draw of all elements by calling _update_plot_data
        # This will also update the legend and regions
        self.state.set_updating(False)  # Ensure not in updating state for first call
        self._update_plot_data(self.ax)
        self.fig.canvas.draw_idle()
        logger.info("Plot rendering complete.")

    def home(self) -> None:
        """Return to initial full view with complete state reset."""
        if self.ax is None:  # Fix: Changed '===' to 'is'
            logger.warning("Plot not rendered yet. Cannot go home.")
            return

        # Disconnect callback temporarily
        callback_id = None
        for cid, callback in self.ax.callbacks.callbacks["xlim_changed"].items():
            if getattr(callback, "__func__", callback) == self._update_plot_data:
                callback_id = cid
                break

        if callback_id is not None:
            self.ax.callbacks.disconnect(callback_id)

        try:
            self.state.set_updating(True)
            self.state.reset_to_initial_state()
            self.decimator.clear_cache()
            self._clear_region_fills()

            # Clear all custom elements and reset _last_mode for each trace to force redraw
            for trace_idx in range(self.data.num_traces):
                self._clear_custom_elements(trace_idx)
                self._last_mode[trace_idx] = None

            # Reset axis formatting
            self.ax.set_xlabel(f"Time ({self.state.original_time_unit})")
            self.ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            self.ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())

            # Reset view
            self.coord_manager.set_view_raw(self.ax, self._initial_xlim_raw)

            # Manually trigger update for the home view
            # This will re-evaluate use_envelope, current_mode, and redraw everything
            self._update_plot_data(self.ax)

            self.state.set_updating(False)

        finally:
            self.ax.callbacks.connect("xlim_changed", self._update_plot_data)

        self.fig.canvas.draw()
        logger.info(f"Home view restored: {self.state.original_time_unit} scale")

    def refresh(self) -> None:
        """Force a complete refresh of the plot without changing the current view."""
        if self.ax is None:
            logger.warning("Plot not rendered yet. Cannot refresh.")
            return

        # Temporarily bypass the updating state for forced refresh
        was_updating = self.state.is_updating()
        self.state.set_updating(False)
        try:
            self._update_plot_data(self.ax)
        finally:
            self.state.set_updating(was_updating)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Display the plot."""
        if self.fig is None:
            self.render()  # Render if not already rendered
        plt.show()
