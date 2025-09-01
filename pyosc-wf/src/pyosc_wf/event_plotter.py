import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from pyosc.oscplot.display_state import (
    _create_time_formatter,
    _determine_offset_display_params,
    _get_optimal_time_unit_and_scale,
)
from pyosc.oscplot.plot import OscilloscopePlot


class EventPlotter:
    """
    Provides utility functions for plotting individual events or event grids.
    """

    def __init__(
        self,
        osc_plot: OscilloscopePlot,
        events: Optional[np.ndarray] = None,
        trace_idx: int = 0,
        bg_clean: Optional[np.ndarray] = None,
        global_noise: Optional[np.float32] = None,
        y_scale_mode: str = "raw",
    ):
        """
        Initialize the EventPlotter with an OscilloscopePlot instance.

        Parameters
        ----------
        osc_plot : OscilloscopePlot
            An instance of OscilloscopePlot containing the waveform data.
        events : Optional[np.ndarray], default=None
            Events array with shape (n_events, 2) where each row is [start_time, end_time].
            If None, will try to extract events from regions in the OscilloscopePlot.
        trace_idx : int, default=0
            Index of the trace to extract events from.
        bg_clean : Optional[np.ndarray], default=None
            The clean background signal array. This is needed for plotting background in event views.
        global_noise : Optional[np.float32], default=None
            The estimated global noise level. If provided, a noise ribbon will be plotted around bg_clean.
        y_scale_mode : str, default="raw"
            Y-axis scaling mode. Options:
            - "raw": Raw signal values
            - "percent": Percentage contrast relative to background ((signal - bg) / bg * 100)
            - "snr": Signal-to-noise ratio ((signal - bg) / noise)
        """
        self.osc_plot = osc_plot
        self.trace_idx = trace_idx
        self.bg_clean = bg_clean
        self.global_noise = global_noise  # Store global_noise here
        self.y_scale_mode = y_scale_mode

        # Extract events from regions if not provided
        if events is None:
            self.events = self._extract_events_from_regions()
        else:
            self.events = events

        if self.events is None or len(self.events) == 0:
            logger.warning("EventPlotter initialized but no events are available.")
            self.events = np.array([])  # Ensure it's an empty array if no events

        # Validate y_scale_mode
        valid_modes = ["raw", "percent", "snr"]
        if self.y_scale_mode not in valid_modes:
            logger.warning(
                f"Invalid y_scale_mode '{self.y_scale_mode}'. Using 'raw'. Valid options: {valid_modes}"
            )
            self.y_scale_mode = "raw"

        # Warn if scaling mode requires data that's not available
        if self.y_scale_mode == "percent" and self.bg_clean is None:
            logger.warning(
                "y_scale_mode='percent' requires bg_clean data. Falling back to 'raw' mode."
            )
            self.y_scale_mode = "raw"
        elif self.y_scale_mode == "snr" and self.global_noise is None:
            logger.warning(
                "y_scale_mode='snr' requires global_noise data. Falling back to 'raw' mode."
            )
            self.y_scale_mode = "raw"

        self.fig: Optional[matplotlib.figure.Figure] = None

    def save(self, filepath: str):
        """
        Save the current state of the EventPlotter to a file.

        Parameters
        ----------
        filepath : str
            Path to save the EventPlotter state.
        """
        if self.fig is not None:
            self.fig.savefig(filepath)
            logger.info(f"EventPlotter figure saved to {filepath}")

    def _extract_events_from_regions(self) -> Optional[np.ndarray]:
        """
        Extract events from regions in the OscilloscopePlot.

        Returns
        -------
        Optional[np.ndarray]
            Events array with shape (n_events, 2) where each row is [start_time, end_time].
        """
        # First check if events are stored in the data manager
        if hasattr(self.osc_plot.data, "get_events"):
            events = self.osc_plot.data.get_events(self.trace_idx)
            if events is not None:
                return events

        # If not, try to extract from regions (backward compatibility)
        if not hasattr(self.osc_plot, "_regions") or not self.osc_plot._regions:
            return None

        # Extract regions from the specified trace
        trace_regions = self.osc_plot._regions[self.trace_idx]
        if not trace_regions:
            return None

        # Combine all regions into a single array
        all_events = []
        for region_def in trace_regions:
            if "regions" in region_def and region_def["regions"] is not None:
                all_events.append(region_def["regions"])

        if not all_events:
            return None

        # Concatenate all region arrays
        return np.vstack(all_events)

    def _scale_y_data(
        self, y_data: np.ndarray, bg_data: Optional[np.ndarray], mask: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """
        Scale y-data according to the current scaling mode.

        Parameters
        ----------
        y_data : np.ndarray
            Raw signal data.
        bg_data : Optional[np.ndarray]
            Background data array (same length as full signal).
        mask : np.ndarray
            Boolean mask for extracting the relevant portion of bg_data.

        Returns
        -------
        Tuple[np.ndarray, str]
            Scaled y-data and appropriate y-axis label.
        """
        if self.y_scale_mode == "percent" and bg_data is not None:
            bg_event = bg_data[mask]
            # Avoid division by zero - use small value for near-zero background
            bg_safe = np.where(np.abs(bg_event) < 1e-12, 1e-12, bg_event)
            scaled_data = 100 * (y_data - bg_event) / bg_safe
            return scaled_data, "Contrast (%)"
        elif self.y_scale_mode == "snr" and self.global_noise is not None:
            bg_event = bg_data[mask] if bg_data is not None else 0
            scaled_data = (y_data - bg_event) / self.global_noise
            return scaled_data, "Signal (σ)"
        else:
            return y_data, "Signal"

    def _scale_background_data(self, bg_data: np.ndarray) -> np.ndarray:
        """
        Scale background data according to the current scaling mode.

        Parameters
        ----------
        bg_data : np.ndarray
            Background data.

        Returns
        -------
        np.ndarray
            Scaled background data.
        """
        if self.y_scale_mode == "percent":
            # In percentage mode, background becomes 0% contrast
            return np.zeros_like(bg_data)
        elif self.y_scale_mode == "snr":
            # In SNR mode, background becomes 0 sigma
            return np.zeros_like(bg_data)
        else:
            return bg_data

    def _scale_noise_ribbon(
        self, bg_data: np.ndarray, noise_level: np.float32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale noise ribbon bounds according to the current scaling mode.

        Parameters
        ----------
        bg_data : np.ndarray
            Background data.
        noise_level : np.float32
            Noise level (±1σ).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds for noise ribbon.
        """
        if self.y_scale_mode == "percent":
            # In percentage mode, noise becomes ±(noise/bg * 100)%
            bg_safe = np.where(np.abs(bg_data) < 1e-12, 1e-12, bg_data)
            noise_percent = 100 * noise_level / np.abs(bg_safe)
            return -noise_percent, noise_percent
        elif self.y_scale_mode == "snr":
            # In SNR mode, noise becomes ±1σ
            return np.full_like(bg_data, -1.0), np.full_like(bg_data, 1.0)
        else:
            return bg_data - noise_level, bg_data + noise_level

    def plot_single_event(self, event_index: int) -> None:
        """
        Plot an individual event.

        Parameters
        ----------
        event_index : int
            Index of the event to plot.
        """
        if self.events is None or len(self.events) == 0:
            logger.warning("No events available to plot.")
            return

        if not (0 <= event_index < len(self.events)):
            logger.warning(
                f"Event index {event_index} out of bounds. Total events: {len(self.events)}."
            )
            return

        t_start, t_end = self.events[event_index]

        # Get the raw data for the specific trace
        t_raw = self.osc_plot.data.t_arrays[self.trace_idx]
        x_raw = self.osc_plot.data.x_arrays[self.trace_idx]

        # Define a window around the event
        event_duration = t_end - t_start
        plot_start_time = t_start - event_duration * 0.5
        plot_end_time = t_end + event_duration * 0.5

        # Get data within the plot window
        mask = (t_raw >= plot_start_time) & (t_raw <= plot_end_time)
        t_event_raw = t_raw[mask]
        x_event = x_raw[mask]

        if not np.any(mask):
            logger.warning(
                f"No data found for event {event_index} in time range [{t_start:.6f}, {t_end:.6f}]"
            )
            return

        # Extract event data and make relative to plot window start
        t_event_raw_relative = t_event_raw - plot_start_time

        # Determine offset display parameters
        time_span_raw = plot_end_time - plot_start_time
        event_time_unit, display_scale, offset_time_raw, offset_unit = (
            _determine_offset_display_params(
                (plot_start_time, plot_end_time), time_span_raw
            )
        )

        # Scale for display using the display_scale from offset params
        t_event_display = t_event_raw_relative * display_scale

        # Create time formatter for axis
        time_formatter = _create_time_formatter(offset_time_raw, display_scale)

        # Scale the data according to the selected mode
        x_event_scaled, ylabel = self._scale_y_data(x_event, self.bg_clean, mask)

        self.fig, ax_ev = plt.subplots(figsize=(6, 3))
        ax_ev.plot(
            t_event_display,
            x_event_scaled,
            label="Event",
            color="black",
            # marker="o",
            mfc="none",
        )

        if self.bg_clean is not None:
            bg_event_scaled = self._scale_background_data(self.bg_clean[mask])
            ax_ev.plot(
                t_event_display,
                bg_event_scaled,
                label="BG",
                color="orange",
                ls="--",
            )
            if self.global_noise is not None:
                # Plot noise ribbon around the background
                noise_lower, noise_upper = self._scale_noise_ribbon(
                    self.bg_clean[mask], self.global_noise
                )
                ax_ev.fill_between(
                    t_event_display,
                    noise_lower,
                    noise_upper,
                    color="gray",
                    alpha=0.3,
                    label="Noise (±1σ)",
                )
        else:
            logger.warning(
                "Clean background (bg_clean) not provided to EventPlotter, cannot plot."
            )

        # Set xlabel with offset if applicable
        if offset_time_raw is not None:
            # Use the offset unit for display, not the event time unit
            offset_scale = 1.0
            if offset_unit == "ms":
                offset_scale = 1e3
            elif offset_unit == "us":
                offset_scale = 1e6
            elif offset_unit == "ns":
                offset_scale = 1e9

            offset_display = offset_time_raw * offset_scale
            ax_ev.set_xlabel(
                f"Time ({event_time_unit}) + {offset_display:.3g} {offset_unit}"
            )
        else:
            ax_ev.set_xlabel(f"Time ({event_time_unit})")

        # Apply the time formatter to x-axis
        ax_ev.xaxis.set_major_formatter(time_formatter)
        # Use a shorter title - just the base trace name without noise info
        trace_name = self.osc_plot.data.get_trace_name(self.trace_idx)
        # Remove noise information from title if present
        clean_name = trace_name.split(" | ")[0] if " | " in trace_name else trace_name
        ax_ev.set_title(f"{clean_name} - Event {event_index + 1}")
        ax_ev.set_ylabel(ylabel)
        ax_ev.legend(loc="lower right")

    def plot_events_grid(self, max_events: int = 16) -> None:
        """
        Plot multiple events in a subplot grid.

        Parameters
        ----------
        max_events : int, default=16
            Maximum number of events to plot in the grid.
        """
        if self.events is None or len(self.events) == 0:
            logger.warning("No events available to plot.")
            return

        # Limit number of events
        n_events = min(len(self.events), max_events)
        events_to_plot = self.events[:n_events]

        # Determine grid size
        if n_events <= 4:
            rows, cols = 2, 2
        elif n_events <= 9:
            rows, cols = 3, 3
        elif n_events <= 16:
            rows, cols = 4, 4
        elif n_events <= 25:
            rows, cols = 5, 5
        else:
            rows, cols = 6, 6  # Maximum 36 events

        self.fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 3), sharey=True
        )  # Sharey for consistent amplitude scale
        # Get trace name safely and clean it
        trace_name = self.osc_plot.data.get_trace_name(self.trace_idx)
        # Remove noise information from title if present
        clean_name = trace_name.split(" | ")[0] if " | " in trace_name else trace_name

        self.fig.suptitle(
            f"{clean_name} - Events 1-{n_events} (of {len(self.events)} total)",
            fontsize=12,
        )

        # Flatten axes for easier indexing
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Get the raw data for the specific trace once
        t_raw_full = self.osc_plot.data.t_arrays[self.trace_idx]
        x_raw_full = self.osc_plot.data.x_arrays[self.trace_idx]

        for i, (t_start, t_end) in enumerate(events_to_plot):
            ax = axes[i]

            # Define a window around the event
            event_duration = t_end - t_start
            plot_start_time = t_start - event_duration * 0.5
            plot_end_time = t_end + event_duration * 0.5

            # Extract event data
            mask = (t_raw_full >= plot_start_time) & (t_raw_full <= plot_end_time)
            t_event_raw = t_raw_full[mask]
            x_event = x_raw_full[mask]

            if not np.any(mask):
                ax.text(
                    0.5,
                    0.5,
                    f"Event {i + 1}\nNo data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Make time relative to plot window start
            t_event_raw_relative = t_event_raw - plot_start_time

            # Determine offset display parameters for this event
            time_span_raw = plot_end_time - plot_start_time
            event_time_unit, display_scale, offset_time_raw, offset_unit = (
                _determine_offset_display_params(
                    (plot_start_time, plot_end_time), time_span_raw
                )
            )

            # Scale for display using the display_scale from offset params
            t_event_display = t_event_raw_relative * display_scale

            # Create time formatter for axis
            time_formatter = _create_time_formatter(offset_time_raw, display_scale)

            # Scale the data according to the selected mode
            x_event_scaled, ylabel = self._scale_y_data(x_event, self.bg_clean, mask)

            # Plot event
            ax.plot(
                t_event_display,
                x_event_scaled,
                "-ok",
                mfc="none",
                linewidth=1,
                label="Signal",
                ms=4,
            )

            if self.bg_clean is not None:
                bg_event_scaled = self._scale_background_data(self.bg_clean[mask])
                ax.plot(
                    t_event_display,
                    bg_event_scaled,
                    "orange",
                    linestyle="--",
                    alpha=0.7,
                    label="BG",
                )
                if self.global_noise is not None:
                    # Plot noise ribbon around the background
                    noise_lower, noise_upper = self._scale_noise_ribbon(
                        self.bg_clean[mask], self.global_noise
                    )
                    ax.fill_between(
                        t_event_display,
                        noise_lower,
                        noise_upper,
                        color="gray",
                        alpha=0.3,
                        label="Noise (±1σ)",
                    )
            else:
                logger.warning(
                    f"Background data not available for event {i + 1}. Ensure bg_clean is passed to EventPlotter."
                )

            # Formatting
            ax.set_title(f"Event {i + 1}", fontsize=10)

            # Set xlabel with offset if applicable
            if offset_time_raw is not None:
                # Use the offset unit for display, not the event time unit
                offset_scale = 1.0
                if offset_unit == "ms":
                    offset_scale = 1e3
                elif offset_unit == "us":
                    offset_scale = 1e6
                elif offset_unit == "ns":
                    offset_scale = 1e9

                offset_display = offset_time_raw * offset_scale
                ax.set_xlabel(
                    f"Time ({event_time_unit}) + {offset_display:.3g} {offset_unit}",
                    fontsize=8,
                )
            else:
                ax.set_xlabel(f"Time ({event_time_unit})", fontsize=8)

            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)

            # Apply the time formatter to x-axis
            ax.xaxis.set_major_formatter(time_formatter)

            # Only show legend on first subplot
            if i == 0:
                ax.legend(fontsize=7, loc="best")

        # Hide unused subplots
        for i in range(n_events, len(axes)):
            axes[i].set_visible(False)
