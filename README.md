# pyosc (transivent)

*Note: The project name `pyosc` is a placeholder and will be changed.*

`transivent` is a Python library for detecting and analysing transient events in time-series data. It provides a flexible and configurable pipeline for processing waveform data, identifying events based on signal-to-noise ratio, and visualizing the results.

## Key Features

-   **Event Detection:** A robust pipeline for detecting transient events based on a configurable signal-to-noise ratio (SNR) threshold against a calculated background.
-   **Configurable Analysis:** Easily configure parameters like smoothing windows, SNR thresholds, event duration, and signal polarity.
-   **Chunked Processing:** Efficiently process files that are too large to fit into memory by reading and analysing them in chunks.
-   **Visualization:** Integrates with `pyosc` to provide interactive plots of the waveform, background, detection thresholds, and detected events.

## Quick Start

The primary entrypoint for analysis is the `transivent.process_file` function. An example of how to configure and run an analysis can be found in `example.py`.

### Example Configuration

The analysis pipeline is controlled via a configuration dictionary.

```python
# From example.py
CONFIG = {
    "SMOOTH_WIN_T": 10e-3,
    "DETECTION_SNR": 3,
    "MIN_EVENT_KEEP_SNR": 5,
    "SIGNAL_POLARITY": 1,
    # ... and more
}
```

### Running the Example

To run the default analysis on the example data:

```bash
python example.py
```

### Enabling Chunked Processing

To process a large file, you can enable chunked processing by setting the `CHUNK_SIZE` parameter in the configuration. When `CHUNK_SIZE` is set, `process_file` will read the data in chunks of the specified size.

```python
# In example.py, change CHUNK_SIZE
"CHUNK_SIZE": 1_000_000,  # Process in chunks of 1 million points
```
