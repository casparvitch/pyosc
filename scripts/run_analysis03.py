from warnings import warn

import matplotlib as mpl

from pyosc.waveform import configure_logging, get_waveform_params, process_file

# --- User configuration dictionary ---
CONFIG = {
    "SMOOTH_WIN_T": 50e-6,  # smoothing window in seconds (set to None to use frequency)
    "SMOOTH_WIN_F": None,  # smoothing window in Hz (set to None to use time)
    "DETECTION_SNR": 4,  # point-by-point detection threshold, <MIN_EVENT_KEEP_SNR
    "MIN_EVENT_KEEP_SNR": 6,  # min event (max-)amplitude in multiples of global noise
    "MIN_EVENT_T": 0.75e-6,  # minimum event duration (seconds)
    "WIDEN_FRAC": 10,  # fraction of event length to widen detected events
    "SIGNAL_POLARITY": -1,  # Signal polarity: -1 for negative events (below background), +1 for positive events (above background)
    "LOG_LEVEL": "SUCCESS",  # logging level: DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
    "MAX_PLOT_POINTS": 10000,  # Downsample threshold for plotting
    "ENVELOPE_MODE_LIMIT": 10e-3,  # Use envelope when time span >10ms, show thresholds when <10ms
    "YSCALE_MODE": "snr",  # y-scale mode for event plotter: 'snr', 'percent' or 'raw'
    "FILTER_TYPE": "gaussian",  # Filter type: "savgol", "gaussian", "moving_average", "median"
    "FILTER_ORDER": 3,  # Order of the savgol filter for smoothing
    # ---
    "DATA_PATH": "../data/20250703-bsa/",
    "MEASUREMENTS": [
        {
            "data": "RefCurve_2025-07-03_3_014156.Wfm.bin",
        },
        {
            "data": "RefCurve_2025-07-03_4_014809.Wfm.bin",
        },
        {
            "data": "RefCurve_2025-07-03_5_015243.Wfm.bin",
        },
    ],
}


def main() -> None:
    """
    Main function to process all measurements.
    """
    # Configure logging
    configure_logging(CONFIG.get("LOG_LEVEL", "INFO"))

    for measurement in CONFIG["MEASUREMENTS"]:
        # Merge global config with measurement-specific overrides
        merged_config = CONFIG.copy()
        merged_config.update(measurement)

        # Extract parameters for process_file
        name = merged_config["data"]
        sidecar = merged_config.get("sidecar")

        if sidecar:
            params = get_waveform_params(
                name, data_path=merged_config["DATA_PATH"], xml_filename=sidecar
            )
            sampling_interval = params["sampling_interval"]
        else:
            params = get_waveform_params(name, data_path=merged_config["DATA_PATH"])
            sampling_interval = params["sampling_interval"]

        # Call with explicit parameters
        process_file(
            name=name,
            sampling_interval=sampling_interval,
            data_path=merged_config["DATA_PATH"],
            smooth_win_t=merged_config.get("SMOOTH_WIN_T"),
            smooth_win_f=merged_config.get("SMOOTH_WIN_F"),
            detection_snr=merged_config.get("DETECTION_SNR", 3.0),
            min_event_keep_snr=merged_config.get("MIN_EVENT_KEEP_SNR", 6.0),
            min_event_t=merged_config.get("MIN_EVENT_T", 0.75e-6),
            widen_frac=merged_config.get("WIDEN_FRAC", 10.0),
            signal_polarity=merged_config.get("SIGNAL_POLARITY", -1),
            max_plot_points=merged_config.get("MAX_PLOT_POINTS", 10000),
            envelope_mode_limit=merged_config.get("ENVELOPE_MODE_LIMIT", 10e-3),
            sidecar=sidecar,
            crop=merged_config.get("crop"),
            yscale_mode=merged_config.get("YSCALE_MODE", "snr"),
            show_plots=True,
            filter_type=merged_config.get("FILTER_TYPE", "gaussian"),
            filter_order=merged_config.get("FILTER_ORDER", 2),
        )


if __name__ == "__main__":
    # Set Matplotlib rcParams directly here
    for optn, val in {
        "backend": "QtAgg",
        "figure.constrained_layout.use": True,
        "figure.dpi": 90,
        # "figure.figsize": (8.5 / 2.55, 6 / 2.55),
        "font.family": ("sans-serif",),
        "font.size": 11,
        "legend.fontsize": "x-small",
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.6,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 1.6,
        "lines.linewidth": 1.8,
        "xtick.labelsize": 10,
        "xtick.major.size": 3,
        "xtick.direction": "in",
        "ytick.labelsize": 10,
        "ytick.direction": "in",
        "ytick.major.size": 3,
        "axes.formatter.useoffset": False,
        "axes.formatter.use_mathtext": True,
        "errorbar.capsize": 3.0,
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.4,
        "xtick.minor.width": 1.1,
        "ytick.major.width": 1.4,
        "ytick.minor.width": 1.1,
        "axes.labelsize": 11,
    }.items():
        if isinstance(val, (list, tuple)):
            val = tuple(val)
        try:
            mpl.rcParams[optn] = val
        except KeyError:
            warn(f"mpl rcparams key '{optn}' not recognised as a valid rc parameter.")
    main()
