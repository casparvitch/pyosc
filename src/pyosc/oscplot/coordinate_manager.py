from typing import Tuple

import numpy as np
from loguru import logger


class CoordinateManager:
    """
    Handles coordinate transformations between raw time and display coordinates.

    Centralises all coordinate conversion logic to prevent inconsistencies.
    """

    def __init__(self, display_state):
        """
        Initialise the coordinate manager.

        Parameters
        ----------
        display_state : DisplayState
            Reference to the display state object.
        """
        self.state = display_state

    def get_current_view_raw(self, ax):
        """Get current view in raw coordinates."""
        try:
            xlim_display = ax.get_xlim()
            logger.debug(f"Converting display xlim {xlim_display} to raw coordinates")

            # Validate display limits
            if not np.isfinite(xlim_display[0]) or not np.isfinite(xlim_display[1]):
                logger.warning(f"Invalid display limits: {xlim_display}")
                # Try to get a valid view from the figure
                if hasattr(ax, "figure") and hasattr(ax.figure, "canvas"):
                    ax.figure.canvas.draw()
                    xlim_display = ax.get_xlim()
                    if not np.isfinite(xlim_display[0]) or not np.isfinite(
                        xlim_display[1]
                    ):
                        # Still invalid, use a default range
                        logger.warning(
                            "Still invalid after redraw, using default range"
                        )
                        xlim_display = (0, 1)

            raw_coords = self.xlim_display_to_raw(xlim_display)
            logger.debug(f"Converted to raw coordinates: {raw_coords}")
            return raw_coords
        except Exception as e:
            logger.exception(f"Error getting current view: {e}")
            # Return a safe default
            return (np.float32(0.0), np.float32(1.0))

    def set_view_raw(self, ax, xlim_raw):
        """Set view using raw coordinates."""
        xlim_display = self.xlim_raw_to_display(xlim_raw)
        ax.set_xlim(xlim_display)

    def raw_to_display(self, t_raw: np.ndarray) -> np.ndarray:
        """Convert raw time to display coordinates."""
        if self.state.offset_time_raw is not None:
            return (t_raw - self.state.offset_time_raw) * self.state.current_time_scale
        else:
            return t_raw * self.state.current_time_scale

    def display_to_raw(self, t_display: np.ndarray) -> np.ndarray:
        """Convert display coordinates to raw time."""
        t_raw = t_display / self.state.current_time_scale
        if self.state.offset_time_raw is not None:
            t_raw += self.state.offset_time_raw

        # Only log for scalar values to avoid excessive output
        if isinstance(t_display, (int, float, np.number)):
            logger.debug(
                f"Converting display time {t_display:.6f} to raw time {t_raw:.6f} (scale={self.state.current_time_scale}, offset={self.state.offset_time_raw})"
            )
        return t_raw

    def xlim_display_to_raw(
        self, xlim_display: Tuple[float, float]
    ) -> Tuple[np.float32, np.float32]:
        """Convert display xlim tuple to raw time coordinates."""
        try:
            # Ensure values are finite
            if not np.isfinite(xlim_display[0]) or not np.isfinite(xlim_display[1]):
                logger.warning(
                    f"Non-finite display limits: {xlim_display}, using defaults"
                )
                return (np.float32(0.0), np.float32(1.0))

            return (
                self.display_to_raw(np.float32(xlim_display[0])),
                self.display_to_raw(np.float32(xlim_display[1])),
            )
        except Exception as e:
            logger.exception(f"Error converting display to raw coordinates: {e}")
            return (np.float32(0.0), np.float32(1.0))

    def xlim_raw_to_display(
        self, xlim_raw: Tuple[np.float32, np.float32]
    ) -> Tuple[np.float32, np.float32]:
        """Convert raw time xlim tuple to display coordinates."""
        return (
            self.raw_to_display(xlim_raw[0]),
            self.raw_to_display(xlim_raw[1]),
        )
