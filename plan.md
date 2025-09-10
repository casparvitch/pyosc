# Design and Implementation Plan for Chunked Processing

This document outlines the plan to refactor the `transivent` library to support the processing of large data files that do not fit into memory.

## 1. High-Level Design Philosophy

The refactoring will adhere to the following principles:

-   **Simplicity and Maintainability:** Prioritise simple, clear, and robust code over complex abstractions. Follow a "functions-first" approach where possible.
-   **Separation of Concerns:** Strictly separate the file I/O logic (reading data from disk) from the analysis logic (signal processing and event detection).
-   **Stateful Functional Approach:** For the analysis, which is inherently stateful when processing in chunks, we will use a functional pattern where a `state` dictionary is explicitly passed between function calls. This avoids a large, monolithic class while still managing the required state cleanly.

## 2. Implementation Plan

The implementation will be divided into two main phases to manage complexity and ensure a stable transition.

---

### **Phase 1: Architectural Refactoring (No Functional Change)**

**Goal:** Restructure the existing code into a state-passing architecture. In this phase, the entire file will still be loaded into memory, but the processing logic will be adapted to use the new stateful functions, paving the way for true chunked processing.

**Implementation Details:**

1.  **Create State-Passing Functions in `transivent/analysis.py`:**
    -   **`initialize_state(config)`:**
        -   This function will take a configuration dictionary as input.
        -   It will create and return a `state` dictionary that will serve as the primary state-management object. The structure will be:
            ```python
            state = {
                "config": config,
                "events": [],  # To store lists of events from each chunk
                "overlap_buffer": {"t": np.array([]), "x": np.array([])}, # For seamless filtering
                "incomplete_event": None, # To handle events spanning chunks
                # Other state variables as needed...
            }
            ```
    -   **`process_chunk(data, state)`:**
        -   This function will contain the core analysis pipeline, moved from the current `process_file` function.
        -   It will take a `data` tuple `(t, x)` and the current `state` dictionary.
        -   It will perform the sequence of operations: `calculate_initial_background` -> `estimate_noise` -> `detect_initial_events` -> `calculate_clean_background` -> `detect_final_events`.
        -   The detected events for the chunk will be appended to `state["events"]`.
        -   It will return the updated `state` dictionary. For this phase, it will also return intermediate results like `bg_clean` and `bg_initial` for plotting.
    -   **`get_final_events(state)`:**
        -   This function will take the final `state` dictionary after all processing is done.
        -   It will concatenate all event arrays stored in `state["events"]` and call `merge_overlapping_events` to produce the final, consolidated list of events.

2.  **Refactor `process_file` in `transivent/analysis.py`:**
    -   The function's signature will remain the same.
    -   The initial `load_data` call will be kept to load the entire dataset.
    -   The main logic will be replaced by the new three-step process:
        1.  `state = initialize_state(config)`
        2.  `state, bg_clean, ... = process_chunk((t, x), state)` (passing the full dataset as one "chunk").
        3.  `final_events = get_final_events(state)`
    -   The rest of the function, which handles plotting and saving results, will be updated to use the `final_events` and other returned variables.

---

### **Phase 2: Implementing True Chunked I/O and Processing**

**Goal:** Implement the file I/O and state management logic required for true chunk-based processing, allowing the analysis of files larger than memory.

**Implementation Details:**

1.  **Create a Chunked Reader in `pywf/io.py`:**
    -   A new generator function, `rd_chunked(filename, chunk_size, ...)`, will be created.
    -   It will first parse the XML sidecar to get waveform parameters (`sampling_interval`, `dtype`, etc.).
    -   It will then open the binary file and read it in blocks of `chunk_size` points.
    -   For each block, it will generate the corresponding time array and `yield` a `(t_chunk, x_chunk)` tuple.

2.  **Enhance State Management in `transivent/analysis.py`:**
    -   The `process_chunk` function will be enhanced to manage state across chunks:
        -   **Overlap Buffer:** Before processing, it will prepend the `overlap_buffer` from the previous `state` to the current chunk's data. The size of this overlap will be determined by the smoothing window (`smooth_n`) to ensure correct filter calculations at chunk boundaries. After processing, it will save the tail of the current chunk's data back into the `state`'s `overlap_buffer` for the next iteration.
        -   **Event Merging:** It will need logic to handle an event that is detected at the very end of a chunk. This "incomplete event" will be stored in the `state`. In the next iteration, if an event is detected at the beginning of the new chunk, the two will be merged.

3.  **Update `process_file` for Streaming:**
    -   The function will be modified to handle a new `chunked=True` mode.
    -   Instead of `load_data`, it will use the `rd_chunked` generator in a `for` loop.
    -   Inside the loop, it will call `process_chunk` for each yielded chunk, passing the `state` dictionary from one iteration to the next.
    -   After the loop, it will call `get_final_events` to get the final results.
    -   Plotting logic will need to be adapted. The main trace plot showing the full signal and background will not be possible in chunked mode. We may disable it or find an alternative visualization strategy for large datasets. Event-specific plots will still be possible.
