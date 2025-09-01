from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger
from numba import njit


@njit
def _decimate_time_numba(t: np.ndarray, step: int, n_bins: int) -> np.ndarray:
    """
    Numba-optimized time decimation using bin centers.

    Parameters
    ----------
    t : np.ndarray
        Input time array.
    step : int
        Step size for binning.
    n_bins : int
        Number of bins to create.

    Returns
    -------
    np.ndarray
        Decimated time array with center time of each bin.
    """
    t_decimated = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        start_idx = i * step
        end_idx = min((i + 1) * step, len(t))
        center_idx = start_idx + (end_idx - start_idx) // 2
        t_decimated[i] = t[center_idx]

    return t_decimated


@njit
def _decimate_mean_numba(x: np.ndarray, step: int, n_bins: int) -> np.ndarray:
    """
    Numba-optimized mean decimation.

    Parameters
    ----------
    x : np.ndarray
        Input signal array.
    step : int
        Step size for binning.
    n_bins : int
        Number of bins to create.

    Returns
    -------
    np.ndarray
        Decimated signal array with mean values.
    """
    x_decimated = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        start_idx = i * step
        end_idx = min((i + 1) * step, len(x))

        if end_idx > start_idx:
            # Calculate mean manually for Numba compatibility
            bin_sum = 0.0
            bin_count = end_idx - start_idx
            for j in range(start_idx, end_idx):
                bin_sum += x[j]
            x_decimated[i] = bin_sum / bin_count
        else:
            x_decimated[i] = x[start_idx] if start_idx < len(x) else 0.0

    return x_decimated


@njit
def _decimate_envelope_standard_numba(
    x: np.ndarray, step: int, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized standard envelope decimation.

    Parameters
    ----------
    x : np.ndarray
        Input signal array.
    step : int
        Step size for binning.
    n_bins : int
        Number of bins to create.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Decimated signal (mean), min envelope, max envelope arrays.
    """
    x_decimated = np.zeros(n_bins, dtype=np.float32)
    x_min_envelope = np.zeros(n_bins, dtype=np.float32)
    x_max_envelope = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        start_idx = i * step
        end_idx = min((i + 1) * step, len(x))

        if end_idx > start_idx:
            # Find min and max manually for Numba compatibility
            bin_min = x[start_idx]
            bin_max = x[start_idx]
            bin_sum = 0.0

            for j in range(start_idx, end_idx):
                val = x[j]
                if val < bin_min:
                    bin_min = val
                if val > bin_max:
                    bin_max = val
                bin_sum += val

            x_min_envelope[i] = bin_min
            x_max_envelope[i] = bin_max
            x_decimated[i] = bin_sum / (end_idx - start_idx)
        else:
            fallback_val = x[start_idx] if start_idx < len(x) else 0.0
            x_min_envelope[i] = fallback_val
            x_max_envelope[i] = fallback_val
            x_decimated[i] = fallback_val

    return x_decimated, x_min_envelope, x_max_envelope


@njit
def _decimate_envelope_highres_numba(
    t: np.ndarray, x: np.ndarray, step: int, n_bins: int, envelope_window_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized high-resolution envelope decimation.

    Parameters
    ----------
    t : np.ndarray
        Input time array.
    x : np.ndarray
        Input signal array.
    step : int
        Step size for binning.
    n_bins : int
        Number of bins to create.
    envelope_window_samples : int
        Window size in samples for high-resolution envelope calculation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Decimated signal (mean), min envelope, max envelope arrays.
    """
    x_decimated = np.zeros(n_bins, dtype=np.float32)
    x_min_envelope = np.zeros(n_bins, dtype=np.float32)
    x_max_envelope = np.zeros(n_bins, dtype=np.float32)

    half_window = envelope_window_samples // 2

    for i in range(n_bins):
        start_idx = i * step
        end_idx = min((i + 1) * step, len(t))
        bin_center = start_idx + (end_idx - start_idx) // 2

        # Define window around bin center
        window_start = max(0, bin_center - half_window)
        window_end = min(len(x), bin_center + half_window)

        if window_end > window_start:
            # Find min and max in window manually for Numba compatibility
            window_min = x[window_start]
            window_max = x[window_start]

            for j in range(window_start, window_end):
                val = x[j]
                if val < window_min:
                    window_min = val
                if val > window_max:
                    window_max = val

            x_min_envelope[i] = window_min
            x_max_envelope[i] = window_max
            x_decimated[i] = (window_min + window_max) / 2.0
        else:
            fallback_val = x[bin_center] if bin_center < len(x) else 0.0
            x_min_envelope[i] = fallback_val
            x_max_envelope[i] = fallback_val
            x_decimated[i] = fallback_val

    return x_decimated, x_min_envelope, x_max_envelope


class DecimationManager:
    """
    Handles data decimation and caching for efficient plotting.

    Manages different decimation strategies and caches results to improve performance.
    Pre-calculates decimated data at load time for faster zooming.
    """

    # Cache and performance constants
    CACHE_MAX_SIZE = 10
    MIN_VISIBLE_RANGE_DEFAULT = 1e-6  # Default if no global noise is provided
    # Threshold for warning about too many points in detail mode
    DETAIL_MODE_POINT_WARNING_THRESHOLD = 100000

    def __init__(self, cache_max_size: int = CACHE_MAX_SIZE):
        """
        Initialise the decimation manager.

        Parameters
        ----------
        cache_max_size : int, default=PlotConstants.CACHE_MAX_SIZE
            Maximum number of cached decimation results.
        """
        self._cache: Dict[str, Tuple[np.ndarray, ...]] = {}
        self._cache_max_size = cache_max_size
        # Stores pre-decimated envelope data for the full dataset for each trace/line
        # Structure: {trace_id: {'t': np.ndarray, 'x_min': np.ndarray, 'x_max': np.ndarray, ...}}
        self._pre_decimated_envelopes: Dict[int, Dict[str, np.ndarray]] = {}

    def _get_cache_key(
        self,
        xlim_raw: Tuple[np.float32, np.float32],
        max_points: int,
        use_envelope: bool,
        trace_id: Optional[int] = None,
    ) -> str:
        """Generate cache key for decimated data."""
        # Round to reasonable precision to improve cache hits
        xlim_rounded = (round(float(xlim_raw[0]), 9), round(float(xlim_raw[1]), 9))

        # Include trace_id in cache key for multi-trace support
        trace_suffix = f"_t{trace_id}" if trace_id is not None else ""

        return f"{xlim_rounded}_{max_points}_{use_envelope}{trace_suffix}"

    def _manage_cache_size(self) -> None:
        """Remove oldest cache entry if cache is full."""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    def clear_cache(self) -> None:
        """Clear the decimation cache."""
        self._cache.clear()
        # Do NOT clear _pre_decimated_envelopes here, as they are persistent for the full dataset

    def _decimate_data(
        self,
        t: np.ndarray,
        x: np.ndarray,
        max_points: int,
        use_envelope: bool = False,
        envelope_window_samples: Optional[int] = None,
        return_envelope_min_max: bool = False,  # New parameter
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Unified decimation for time and multiple data arrays.

        Parameters
        ----------
        t : np.ndarray
            Time array.
        x : np.ndarray
            Signal array.
        max_points : int, default=5000
            Maximum number of points to display.
        use_envelope : bool, default=False
            Whether to use envelope decimation for the signal array.
        envelope_window_samples : Optional[int], default=None
            Window size in samples for high-resolution envelope calculation.
        return_envelope_min_max : bool, default=False
            If True, returns x_min_envelope and x_max_envelope. Otherwise, returns None for them.
            If None, uses simple binning approach.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
            Decimated time, signal, signal min envelope, signal max envelope arrays.
        """
        # If input arrays are empty, return empty arrays immediately
        if len(t) == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                None,
                None,
            )

        # If not using envelope, always return raw data for the view
        if (
            not use_envelope and not return_envelope_min_max
        ):  # If not using envelope and not explicitly asking for min/max
            return t, x, None, None  # No min/max envelope for raw data

        # If using envelope and data is small enough, return raw data as envelope
        if use_envelope and len(t) <= max_points and return_envelope_min_max:
            return t, x, x, x  # x,x for min/max when no decimation

        # Calculate step size for decimation based on max_points
        step = max(1, len(t) // max_points)

        # For envelope mode, calculate adaptive envelope window based on data density
        adaptive_envelope_window = None
        if use_envelope and len(t) > max_points:
            # Calculate envelope window based on how much we're decimating
            # This ensures envelope resolution matches display capability
            adaptive_envelope_window = max(
                1, step // 2
            )  # Half the step size for smoother envelope
            logger.debug(
                f"Calculated adaptive envelope window: {adaptive_envelope_window} samples (step={step})"
            )

        # Ensure step is not zero, and calculate number of bins
        if step == 0:  # Should not happen with max(1, ...) but as a safeguard
            step = 1
        n_bins = len(t) // step
        if (
            n_bins == 0
        ):  # If data is too short for the calculated step, take at least one bin
            n_bins = 1
            step = len(t)  # Take all points in one bin

        # Ensure arrays are contiguous and correct dtype for Numba
        t_contiguous = np.ascontiguousarray(t, dtype=np.float32)
        x_contiguous = np.ascontiguousarray(x, dtype=np.float32)

        # Decimate time array using Numba-optimized function
        t_decimated = _decimate_time_numba(t_contiguous, step, n_bins)

        # Decimate signal (x) using appropriate Numba-optimized function
        x_min_envelope: Optional[np.ndarray] = None
        x_max_envelope: Optional[np.ndarray] = None

        if use_envelope:  # This block handles the decimation logic (mean or envelope)
            if adaptive_envelope_window is not None and adaptive_envelope_window > 1:
                logger.debug(
                    f"Using adaptive high-resolution envelope with window size {adaptive_envelope_window} samples"
                )

                # Use Numba-optimized high-resolution envelope decimation with adaptive window
                x_decimated, x_min_envelope, x_max_envelope = (
                    _decimate_envelope_highres_numba(
                        t_contiguous,
                        x_contiguous,
                        step,
                        n_bins,
                        adaptive_envelope_window,
                    )
                )

                envelope_thickness = np.mean(x_max_envelope - x_min_envelope)
                logger.debug(
                    f"Adaptive envelope thickness: mean={envelope_thickness:.3g}, min={np.min(x_max_envelope - x_min_envelope):.3g}, max={np.max(x_max_envelope - x_min_envelope):.3g}"
                )
            else:
                logger.debug("Using standard bin-based envelope")

                # Use Numba-optimized standard envelope decimation
                x_decimated, x_min_envelope, x_max_envelope = (
                    _decimate_envelope_standard_numba(x_contiguous, step, n_bins)
                )

                # If we are not returning min/max, then x_decimated should be the mean
                # Otherwise, x_decimated is just the mean of the envelope for internal use
                if not return_envelope_min_max:
                    x_decimated = (x_min_envelope + x_max_envelope) / 2
        else:  # This block is now reached if use_envelope is False AND len(t) > max_points
            logger.debug("Using mean decimation for single line")

            # Use Numba-optimized mean decimation
            x_decimated = _decimate_mean_numba(x_contiguous, step, n_bins)

        # If return_envelope_min_max is False, ensure min/max are None
        if not return_envelope_min_max:
            x_min_envelope = None
            x_max_envelope = None

        return t_decimated, x_decimated, x_min_envelope, x_max_envelope

    def pre_decimate_data(
        self,
        data_id: int,  # Changed from trace_id to data_id to be more generic for custom lines
        t: np.ndarray,
        x: np.ndarray,
        max_points: int,
        envelope_window_samples: Optional[int] = None,  # This parameter is now ignored
    ) -> None:
        """
        Pre-calculate decimated envelope data for the full dataset.
        This is used for fast rendering in zoomed-out (envelope) mode.

        Parameters
        ----------
        data_id : int
            Unique identifier for this data set (e.g., trace_id or custom line ID).
        t : np.ndarray
            Time array (raw time in seconds).
        x : np.ndarray
            Signal array.
        max_points : int
            Maximum number of points for the pre-decimated data.
        envelope_window_samples : Optional[int], default=None
            Window size in samples for high-resolution envelope calculation.
            This will primarily determine the bin size for pre-decimation.
        """
        if len(t) <= max_points:
            # For small datasets, just store the original data as the "pre-decimated" envelope
            # (min/max will be the same as x)
            self._pre_decimated_envelopes[data_id] = {
                "t": t,
                "x": x,  # Store mean/center for consistency
                "x_min": x,
                "x_max": x,
            }
            logger.debug(
                f"Data ID {data_id} is small enough, storing raw as pre-decimated envelope."
            )
            return

        logger.debug(
            f"Pre-decimating data for ID {data_id} to {max_points} points for envelope view."
        )
        # Perform the decimation using the _decimate_data method
        # We force use_envelope=True here for pre-decimation to capture min/max
        # envelope_window_samples is now calculated automatically based on max_points
        t_decimated, x_decimated, x_min, x_max = self._decimate_data(
            t,
            x,
            max_points=max_points,
            use_envelope=True,  # Always pre-decimate with envelope
            envelope_window_samples=None,  # Let _decimate_data calculate adaptive window
            return_envelope_min_max=True,  # Pre-decimation always stores min/max
        )

        # Store pre-decimated envelope data
        self._pre_decimated_envelopes[data_id] = {
            "t": t_decimated,
            "x": x_decimated,  # This is the mean/center of the envelope
            "x_min": x_min,
            "x_max": x_max,
        }

        logger.debug(
            f"Pre-decimated envelope calculated for ID {data_id}: {len(t_decimated)} points."
        )

    def decimate_for_view(
        self,
        t_raw_full: np.ndarray,  # Full resolution time array
        x_raw_full: np.ndarray,  # Full resolution signal array
        xlim_raw: Tuple[np.float32, np.float32],
        max_points: int,
        use_envelope: bool = False,
        data_id: Optional[int] = None,  # Changed from trace_id to data_id
        envelope_window_samples: Optional[int] = None,  # This parameter is now ignored
        mode_switch_threshold: Optional[
            float
        ] = None,  # New parameter for mode switching
        return_envelope_min_max: bool = False,  # New parameter
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Intelligently decimate data for current view with optional envelope mode.

        Parameters
        ----------
        t_raw_full : np.ndarray
            Full resolution time array (raw time in seconds).
        x_raw_full : np.ndarray
            Full resolution signal array.
        xlim_raw : Tuple[np.float32, np.float32]
            Current x-axis limits in raw time (seconds).
        max_points : int
            Maximum number of points to display.
        use_envelope : bool, default=False
            Whether the current display mode is envelope.
        data_id : Optional[int], default=None
            Unique identifier for this data set (e.g., trace_id or custom line ID).
            Used to retrieve pre-decimated envelope data.
        envelope_window_samples : Optional[int], default=None
            Window size in samples for high-resolution envelope calculation.
        return_envelope_min_max : bool, default=False
            If True, returns x_min_envelope and x_max_envelope. Otherwise, returns None for them.
        mode_switch_threshold : Optional[float], default=None
            Time span threshold for switching between envelope and detail modes.
            Used to decide whether to use pre-decimated envelope data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
            Decimated time, signal, signal min envelope, signal max envelope arrays (all in raw time).
        """
        logger.debug(f"=== DecimationManager.decimate_for_view data_id={data_id} ===")
        logger.debug(f"xlim_raw: {xlim_raw}")
        logger.debug(f"use_envelope (requested): {use_envelope}")
        logger.debug(f"max_points: {max_points}")
        logger.debug(
            f"Input data range: t=[{np.min(t_raw_full):.6f}, {np.max(t_raw_full):.6f}], x=[{np.min(x_raw_full):.6f}, {np.max(x_raw_full):.6f}]"
        )

        # Ensure xlim_raw values are valid
        if (
            not np.isfinite(xlim_raw[0])
            or not np.isfinite(xlim_raw[1])
            or xlim_raw[0] == xlim_raw[1]
        ):
            logger.warning(
                f"Invalid xlim_raw values: {xlim_raw}. Using full data range."
            )
            xlim_raw = (np.min(t_raw_full), np.max(t_raw_full))

        # Ensure xlim_raw is in ascending order
        if xlim_raw[0] > xlim_raw[1]:
            logger.warning(f"xlim_raw values out of order: {xlim_raw}. Swapping.")
            xlim_raw = (xlim_raw[1], xlim_raw[0])

        # Calculate current view span
        current_view_span = xlim_raw[1] - xlim_raw[0]

        # Check cache first
        cache_key = self._get_cache_key(
            xlim_raw, max_points, use_envelope, data_id
        )  # Cache key doesn't need return_envelope_min_max
        if cache_key in self._cache:
            logger.debug(f"Using cached decimation for key: {cache_key}")
            return self._cache[cache_key]

        # --- Strategy: Use pre-decimated envelope if in envelope mode and view is wide ---
        if (
            use_envelope
            and data_id is not None
            and data_id in self._pre_decimated_envelopes
        ):
            pre_dec_data = self._pre_decimated_envelopes[data_id]
            pre_dec_t = pre_dec_data["t"]

            if len(pre_dec_t) > 1:
                pre_dec_span = pre_dec_t[-1] - pre_dec_t[0]

                # Calculate how much detail we would gain by re-decimating
                # Find indices for current view in pre-decimated time
                mask = (pre_dec_t >= xlim_raw[0]) & (pre_dec_t <= xlim_raw[1])
                pre_dec_points_in_view = np.sum(mask)

                # Estimate how many points we would get from dynamic decimation
                t_view_mask = (t_raw_full >= xlim_raw[0]) & (t_raw_full <= xlim_raw[1])
                raw_points_in_view = np.sum(t_view_mask)
                potential_decimated_points = min(raw_points_in_view, max_points)

                # Use pre-decimated data only if:
                # 1. Current view span is very large (> 2x mode_switch_threshold), AND
                # 2. Pre-decimated data provides reasonable detail (> max_points/4), AND
                # 3. We wouldn't gain much detail from re-decimating (< 2x improvement)
                use_pre_decimated = (
                    mode_switch_threshold is not None
                    and current_view_span >= 2 * mode_switch_threshold
                    and pre_dec_points_in_view > max_points // 4
                    and potential_decimated_points < 2 * pre_dec_points_in_view
                )

                if use_pre_decimated and np.any(mask):
                    logger.debug(
                        f"Using pre-decimated data for ID {data_id} (envelope mode, very wide view, {pre_dec_points_in_view} points, return_envelope_min_max={return_envelope_min_max})."
                    )

                    # If we need min/max, return them. Otherwise, return None.
                    x_min_ret = (
                        pre_dec_data["x_min"][mask] if return_envelope_min_max else None
                    )
                    x_max_ret = (
                        pre_dec_data["x_max"][mask] if return_envelope_min_max else None
                    )

                    result = (
                        pre_dec_t[mask],
                        pre_dec_data["x"][mask],  # Center of envelope
                        x_min_ret,
                        x_max_ret,
                    )
                    self._manage_cache_size()
                    self._cache[cache_key] = result
                    return result
                else:
                    logger.debug(
                        f"Re-decimating for better detail: view_span={current_view_span:.3e}, pre_dec_points={pre_dec_points_in_view}, potential_points={potential_decimated_points}"
                    )
            else:
                logger.debug(
                    f"Pre-decimated data for ID {data_id} has only one point, falling back to dynamic decimation."
                )
        else:
            logger.debug(
                f"Not using pre-decimated envelope for ID {data_id} (use_envelope={use_envelope}, data_id={data_id in self._pre_decimated_envelopes})."
            )

        # --- Fallback: Dynamic decimation from raw data ---
        logger.debug("Performing dynamic decimation from raw data.")

        # ADDED DEBUG LOGS
        logger.debug(
            f"  t_raw_full min/max: {t_raw_full.min():.6f}, {t_raw_full.max():.6f}"
        )
        logger.debug(f"  xlim_raw: {xlim_raw[0]:.6f}, {xlim_raw[1]:.6f}")

        # Find indices for current view in raw time
        mask = (t_raw_full >= xlim_raw[0]) & (t_raw_full <= xlim_raw[1])

        # ADDED DEBUG LOG
        logger.debug(f"  Mask result: {np.sum(mask)} points selected.")

        if not np.any(mask):
            logger.warning(
                f"No data in view for xlim_raw: {xlim_raw}. Returning empty arrays."
            )
            empty_result = (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                None,
                None,
            )
            # Cache empty result for this view
            self._manage_cache_size()
            self._cache[cache_key] = empty_result
            return empty_result

        t_view = t_raw_full[mask]
        x_view = x_raw_full[mask]

        # Add warning for large number of points in detail mode
        if not use_envelope and len(t_view) > self.DETAIL_MODE_POINT_WARNING_THRESHOLD:
            logger.warning(
                f"Plotting {len(t_view)} points in detail mode. "
                f"Performance may be affected. Consider zooming in further."
            )

        # Use unified decimation approach
        # envelope_window_samples is now calculated automatically based on max_points and data density
        result = self._decimate_data(
            t_view,
            x_view,
            max_points=max_points,
            use_envelope=use_envelope,  # Use requested envelope mode for dynamic decimation
            envelope_window_samples=None,  # Let _decimate_data calculate adaptive window
            return_envelope_min_max=return_envelope_min_max,  # Pass through
        )

        # Cache the result (manage cache size)
        self._manage_cache_size()
        self._cache[cache_key] = result

        # Log the final result
        t_result, x_result, x_min_result, x_max_result = result
        logger.debug(f"Returning result: t len={len(t_result)}, x len={len(x_result)}")
        logger.debug(
            f"Result ranges: t=[{np.min(t_result) if len(t_result) > 0 else 'empty':.6f}, {np.max(t_result) if len(t_result) > 0 else 'empty':.6f}], x=[{np.min(x_result) if len(x_result) > 0 else 'empty':.6f}, {np.max(x_result) if len(x_result) > 0 else 'empty':.6f}]"
        )
        logger.debug(
            f"Envelope: x_min={'None' if x_min_result is None else f'len={len(x_min_result)}'}, x_max={'None' if x_max_result is None else f'len={len(x_max_result)}'}"
        )

        return result
