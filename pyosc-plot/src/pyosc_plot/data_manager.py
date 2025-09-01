from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger


class TimeSeriesDataManager:
    """
    Manages time series data storage and basic operations.

    Handles raw data storage, time scaling, and basic data access patterns.
    It can also store optional associated data like background estimates,
    global noise, and overlay lines.

    Supports multiple traces with shared time axis or individual time axes.
    """

    def __init__(
        self,
        t: Union[np.ndarray, List[np.ndarray]],
        x: Union[np.ndarray, List[np.ndarray]],
        name: Union[str, List[str]] = "Time Series",
        trace_colors: Optional[List[str]] = None,
    ):
        """
        Initialise the data manager.

        Parameters
        ----------
        t : Union[np.ndarray, List[np.ndarray]]
            Time array(s) (raw time in seconds). Can be a single array shared by all traces
            or a list of arrays, one per trace.
        x : Union[np.ndarray, List[np.ndarray]]
            Signal array(s). If t is a single array, x can be a 2D array (traces x samples)
            or a list of 1D arrays. If t is a list, x must be a list of equal length.
        name : Union[str, List[str]], default="Time Series"
            Name(s) for identification. Can be a single string or a list of strings.
        trace_colors : Optional[List[str]], default=None
            Colors for each trace. If None, default colors will be used.

        Raises
        ------
        ValueError
            If input arrays have mismatched lengths or time array is not monotonic.
        """
        # Convert inputs to standardized format: lists of arrays
        self.t_arrays, self.x_arrays, self.names, self.colors = (
            self._standardize_inputs(t, x, name, trace_colors)
        )

        # Validate all data
        for i, (t_arr, x_arr) in enumerate(zip(self.t_arrays, self.x_arrays)):
            self._validate_core_data(t_arr, x_arr, trace_idx=i)

        # Optional associated data (per trace)
        self._overlay_lines: List[List[Dict[str, Any]]] = [
            [] for _ in range(len(self.t_arrays))
        ]

        # For backward compatibility
        if len(self.t_arrays) > 0:
            self.t = self.t_arrays[0]  # Primary time array
            self.x = self.x_arrays[0]  # Primary signal array
            self.name = self.names[0]  # Primary name

    def _standardize_inputs(
        self,
        t: Union[np.ndarray, List[np.ndarray]],
        x: Union[np.ndarray, List[np.ndarray]],
        name: Union[str, List[str]],
        trace_colors: Optional[List[str]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
        """
        Standardize inputs to lists of arrays.

        Parameters
        ----------
        t : Union[np.ndarray, List[np.ndarray]]
            Time array(s).
        x : Union[np.ndarray, List[np.ndarray]]
            Signal array(s).
        name : Union[str, List[str]]
            Name(s) for identification.
        trace_colors : Optional[List[str]]
            Colors for each trace.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]
            Standardized lists of time arrays, signal arrays, names, and colors.
        """
        # Default colors for traces
        default_colors = [
            "black",
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "olive",
        ]

        # Handle time arrays
        if isinstance(t, list):
            t_arrays = [np.asarray(t_arr, dtype=np.float32) for t_arr in t]
            n_traces = len(t_arrays)
        else:
            t_arr = np.asarray(t, dtype=np.float32)

            # Check if x is 2D array or list
            if isinstance(x, list):
                n_traces = len(x)
                t_arrays = [t_arr.copy() for _ in range(n_traces)]
            elif x.ndim == 2:
                n_traces = x.shape[0]
                t_arrays = [t_arr.copy() for _ in range(n_traces)]
            else:
                n_traces = 1
                t_arrays = [t_arr]

        # Handle signal arrays
        if isinstance(x, list):
            if len(x) != n_traces:
                raise ValueError(
                    f"Number of signal arrays ({len(x)}) must match number of time arrays ({n_traces})"
                )
            x_arrays = [np.asarray(x_arr, dtype=np.float32) for x_arr in x]
        elif x.ndim == 2:
            if x.shape[0] != n_traces:
                raise ValueError(
                    f"First dimension of 2D signal array ({x.shape[0]}) must match number of time arrays ({n_traces})"
                )
            x_arrays = [np.asarray(x[i], dtype=np.float32) for i in range(n_traces)]
        else:
            if n_traces != 1:
                raise ValueError(
                    f"Single signal array provided but expected {n_traces} arrays"
                )
            x_arrays = [np.asarray(x, dtype=np.float32)]

        # Handle names
        if isinstance(name, list):
            if len(name) != n_traces:
                logger.warning(
                    f"Number of names ({len(name)}) doesn't match number of traces ({n_traces}). Using defaults."
                )
                names = [f"Trace {i + 1}" for i in range(n_traces)]
            else:
                names = name
        else:
            if n_traces == 1:
                names = [name]
            else:
                if (
                    name == "Time Series"
                ):  # Only use default naming if the default name was used
                    names = [f"Trace {i + 1}" for i in range(n_traces)]
                else:
                    names = [f"{name} {i + 1}" for i in range(n_traces)]

        # Handle colors
        if trace_colors is not None:
            if len(trace_colors) < n_traces:
                logger.warning(
                    f"Not enough colors provided ({len(trace_colors)}). Using defaults for remaining traces."
                )
                colors = trace_colors + [
                    default_colors[i % len(default_colors)]
                    for i in range(len(trace_colors), n_traces)
                ]
            else:
                colors = trace_colors[:n_traces]
        else:
            colors = [default_colors[i % len(default_colors)] for i in range(n_traces)]

        return t_arrays, x_arrays, names, colors

    def _validate_core_data(
        self, t: np.ndarray, x: np.ndarray, trace_idx: int = 0
    ) -> None:
        """
        Validate core input data arrays for consistency and correctness.

        Parameters
        ----------
        t : np.ndarray
            Time array.
        x : np.ndarray
            Signal array.
        trace_idx : int, default=0
            Index of the trace being validated (for error messages).

        Raises
        ------
        ValueError
            If arrays have mismatched lengths or time array is not monotonic.
        """
        if len(t) != len(x):
            raise ValueError(
                f"Time and signal arrays for trace {trace_idx} must have the same length. Got t={len(t)}, x={len(x)}"
            )
        if len(t) == 0:
            logger.warning(f"Initialising trace {trace_idx} with empty arrays.")
            return

        # Check time array is monotonic
        if len(t) > 1:
            # Use a small epsilon for floating-point comparison
            tolerance = 1e-9
            if not np.all(np.diff(t) > tolerance):
                problematic_diffs = np.diff(t)[np.diff(t) <= tolerance]
                logger.warning(
                    f"Time array for trace {trace_idx} is not strictly monotonic increasing within tolerance {tolerance}. "
                    f"Problematic diffs (first 10): {problematic_diffs[:10]}. "
                    f"This may affect analysis results."
                )

            # Check for non-uniform sampling
            self._check_uniform_sampling(t, trace_idx)

    @property
    def overlay_lines(self) -> List[Dict[str, Any]]:
        """Get overlay lines data for the primary trace."""
        return self._overlay_lines[0] if self._overlay_lines else []

    def get_overlay_lines(self, trace_idx: int = 0) -> List[Dict[str, Any]]:
        """Get overlay lines data for a specific trace."""
        if trace_idx < 0 or trace_idx >= len(self.t_arrays):
            raise ValueError(
                f"Invalid trace index: {trace_idx}. Must be between 0 and {len(self.t_arrays) - 1}."
            )
        return self._overlay_lines[trace_idx]

    @property
    def num_traces(self) -> int:
        """Get the number of traces."""
        return len(self.t_arrays)

    def get_trace_color(self, trace_idx: int = 0) -> str:
        """Get the color for a specific trace."""
        if trace_idx < 0 or trace_idx >= len(self.t_arrays):
            raise ValueError(
                f"Invalid trace index: {trace_idx}. Must be between 0 and {len(self.t_arrays) - 1}."
            )
        return self.colors[trace_idx]

    def get_trace_name(self, trace_idx: int = 0) -> str:
        """Get the name for a specific trace."""
        if trace_idx < 0 or trace_idx >= len(self.t_arrays):
            raise ValueError(
                f"Invalid trace index: {trace_idx}. Must be between 0 and {len(self.t_arrays) - 1}."
            )
        return self.names[trace_idx]

    def set_overlay_lines(
        self,
        overlay_lines: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        trace_idx: Optional[int] = None,
    ) -> None:
        """
        Set overlay lines data.

        Parameters
        ----------
        overlay_lines : Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
            List of dictionaries defining overlay lines, or list of lists for multiple traces.
        trace_idx : Optional[int], default=None
            If provided, set overlay lines only for the specified trace.
            If None, set for all traces if a nested list is provided, or for the first trace if a flat list.
        """
        if trace_idx is not None:
            # Set for specific trace
            if trace_idx < 0 or trace_idx >= len(self.t_arrays):
                raise ValueError(
                    f"Invalid trace index: {trace_idx}. Must be between 0 and {len(self.t_arrays) - 1}."
                )

            # Ensure we have a list of dictionaries
            if not isinstance(overlay_lines, list):
                raise ValueError(
                    f"overlay_lines must be a list of dictionaries. Got {type(overlay_lines)}."
                )

            # Check if it's a list of dictionaries (not a nested list)
            if len(overlay_lines) > 0 and isinstance(overlay_lines[0], dict):
                self._overlay_lines[trace_idx] = overlay_lines
            else:
                raise ValueError(
                    "Expected a list of dictionaries for overlay_lines when trace_idx is specified."
                )
        else:
            # Set for all traces or first trace
            if len(overlay_lines) > 0 and isinstance(overlay_lines[0], list):
                # Nested list provided - set for multiple traces
                if len(overlay_lines) != len(self.t_arrays):
                    raise ValueError(
                        f"Number of overlay line lists ({len(overlay_lines)}) must match number of traces ({len(self.t_arrays)})."
                    )

                for i, lines in enumerate(overlay_lines):
                    self._overlay_lines[i] = lines
            else:
                # Flat list provided - set for first trace
                self._overlay_lines[0] = overlay_lines

    def get_time_range(self, trace_idx: int = 0) -> Tuple[np.float32, np.float32]:
        """
        Get the full time range of the data.

        Parameters
        ----------
        trace_idx : int, default=0
            Index of the trace to get the time range for.

        Returns
        -------
        Tuple[np.float32, np.float32]
            Start and end time of the data.
        """
        if trace_idx < 0 or trace_idx >= len(self.t_arrays):
            raise ValueError(
                f"Invalid trace index: {trace_idx}. Must be between 0 and {len(self.t_arrays) - 1}."
            )

        t_arr = self.t_arrays[trace_idx]
        if t_arr.size == 0:
            return np.float32(0.0), np.float32(0.0)
        return np.float32(t_arr[0]), np.float32(t_arr[-1])

    def get_global_time_range(self) -> Tuple[np.float32, np.float32]:
        """
        Get the global time range across all traces.

        Returns
        -------
        Tuple[np.float32, np.float32]
            Global start and end time across all traces.
        """
        if len(self.t_arrays) == 0:
            return np.float32(0.0), np.float32(0.0)

        t_min = np.float32(
            min(t_arr[0] if t_arr.size > 0 else np.inf for t_arr in self.t_arrays)
        )
        t_max = np.float32(
            max(t_arr[-1] if t_arr.size > 0 else -np.inf for t_arr in self.t_arrays)
        )

        if np.isinf(t_min) or np.isinf(t_max):
            return np.float32(0.0), np.float32(0.0)

        return t_min, t_max

    def get_data_in_range(
        self, t_start: np.float32, t_end: np.float32, trace_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract data within a time range.

        Parameters
        ----------
        t_start : np.float32
            Start time in raw seconds.
        t_end : np.float32
            End time in raw seconds.
        trace_idx : int, default=0
            Index of the trace to get data for.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Time and signal arrays.
        """
        if trace_idx < 0 or trace_idx >= len(self.t_arrays):
            raise ValueError(
                f"Invalid trace index: {trace_idx}. Must be between 0 and {len(self.t_arrays) - 1}."
            )

        t_arr = self.t_arrays[trace_idx]
        x_arr = self.x_arrays[trace_idx]

        mask = (t_arr >= t_start) & (t_arr <= t_end)
        if not np.any(mask):
            logger.debug(f"No data in range [{t_start}, {t_end}] for trace {trace_idx}")
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        t_masked = t_arr[mask]
        x_masked = x_arr[mask]

        return t_masked, x_masked

    def _check_uniform_sampling(self, t: np.ndarray, trace_idx: int = 0) -> None:
        """
        Check if time array is uniformly sampled and issue warnings if not.

        Parameters
        ----------
        t : np.ndarray
            Time array to check.
        trace_idx : int, default=0
            Index of the trace being checked (for warning messages).
        """
        if len(t) < 3:
            return  # Not enough points to check uniformity

        # Calculate time differences
        dt = np.diff(t)

        # Calculate statistics
        dt_mean = np.mean(dt)
        dt_std = np.std(dt)
        dt_cv = dt_std / dt_mean if dt_mean > 0 else 0  # Coefficient of variation

        # Check for significant non-uniformity
        # CV > 0.01 (1%) indicates potentially problematic non-uniformity
        if dt_cv > 0.01:
            logger.warning(
                f"Non-uniform sampling detected in trace {trace_idx}: "
                f"mean dt={dt_mean:.3e}s, std={dt_std:.3e}s, CV={dt_cv:.2%}"
            )

            # More detailed warning for severe non-uniformity
            if dt_cv > 0.05:  # 5% variation
                # Find the most extreme deviations
                dt_median = np.median(dt)
                rel_deviations = np.abs(dt - dt_median) / dt_median
                worst_indices = np.argsort(rel_deviations)[-5:]  # 5 worst points

                worst_deviations = []
                for idx in reversed(worst_indices):
                    if (
                        rel_deviations[idx] > 0.1
                    ):  # Only report significant deviations (>10%)
                        worst_deviations.append(
                            f"at t={t[idx]:.3e}s: dt={dt[idx]:.3e}s ({rel_deviations[idx]:.1%} deviation)"
                        )

                if worst_deviations:
                    logger.warning(
                        f"Severe sampling irregularities detected in trace {trace_idx}. "
                        f"Worst points: {'; '.join(worst_deviations)}"
                    )
                    logger.warning(
                        "Non-uniform sampling may affect analysis results, especially for "
                        "frequency-domain analysis or event detection."
                    )
