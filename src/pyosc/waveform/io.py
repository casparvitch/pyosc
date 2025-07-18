import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
from loguru import logger


def _get_xml_sidecar_path(
    bin_filename: str, 
    data_path: Optional[str] = None, 
    sidecar: Optional[str] = None
) -> str:
    """
    Determine the XML sidecar file path using consistent logic.
    
    Parameters
    ----------
    bin_filename : str
        Name of the binary waveform file.
    data_path : str, optional
        Path to the data directory.
    sidecar : str, optional
        Name of the XML sidecar file. If None, auto-detects from bin_filename.
        
    Returns
    -------
    str
        Full path to the XML sidecar file.
    """
    if sidecar is not None:
        sidecar_path = (
            os.path.join(data_path, sidecar)
            if data_path is not None and not os.path.isabs(sidecar)
            else sidecar
        )
    else:
        base = os.path.splitext(bin_filename)[0]
        if base.endswith(".Wfm"):
            sidecar_guess = base[:-4] + ".bin"
        else:
            sidecar_guess = base + ".bin"
        sidecar_path = (
            os.path.join(data_path, sidecar_guess)
            if data_path is not None
            else sidecar_guess
        )
    return sidecar_path


def get_waveform_params(
    bin_filename: str,
    data_path: Optional[str] = None,
    sidecar: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parse XML sidecar file to extract waveform parameters.

    Given a binary waveform filename, find and parse the corresponding XML sidecar file.
    If sidecar is provided, use it directly. Otherwise, guess from bin_filename.

    Parameters
    ----------
    bin_filename : str
        Name of the binary waveform file.
    data_path : str, optional
        Path to the data directory. If None, uses current directory.
    sidecar : str, optional
        Name of the XML sidecar file. If None, guesses from bin_filename.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: sampling_interval, vertical_scale, vertical_offset,
        byte_order, signal_format.

    Raises
    ------
    FileNotFoundError
        If the XML sidecar file is not found.
    RuntimeError
        If the XML file cannot be parsed.

    Warns
    -----
    UserWarning
        If sampling resolution is not found in XML.
    """
    sidecar_path = _get_xml_sidecar_path(bin_filename, data_path, sidecar)
    params = {
        "sampling_interval": None,
        "vertical_scale": None,
        "vertical_offset": None,
        "byte_order": "LSB",  # default
        "signal_format": "float32",  # default
        "signal_hardware_record_length": None,
    }
    found_resolution = False
    if not os.path.exists(sidecar_path):
        msg = (
            f"XML sidecar file not found: {sidecar_path}\n"
            f"  bin_filename: {bin_filename}\n"
            f"  sidecar: {sidecar}\n"
            f"  data_path: {data_path}\n"
            f"  Tried path: {sidecar_path}\n"
            f"Please check that the XML sidecar exists and the path is correct."
        )
        raise FileNotFoundError(msg)
    try:
        tree = ET.parse(sidecar_path)
        root = tree.getroot()

        # Validate XML structure
        if root is None:
            raise RuntimeError(f"XML file has no root element: {sidecar_path}")

        # Track which parameters we found for validation
        found_params = set()
        signal_resolution = None
        resolution = None

        for prop in root.iter("Prop"):
            if prop.attrib is None:
                logger.warning(f"Found Prop element with no attributes in {sidecar_path}")
                continue

            name = prop.attrib.get("Name", "")
            value = prop.attrib.get("Value", "")

            if not name:
                logger.warning(
                    f"Found Prop element with empty Name attribute in {sidecar_path}"
                )
                continue

            try:
                if name == "Resolution":
                    params["sampling_interval"] = float(value)
                    found_resolution = True
                    found_params.add("SignalResolution")
                    resolution = float(value)
                elif name == "SignalResolution" and params["sampling_interval"] is None:
                    params["sampling_interval"] = float(value)
                    found_resolution = True
                    found_params.add("Resolution")
                    signal_resolution = float(value)
                elif name == "SignalResolution":
                    signal_resolution = float(value)
                elif name == "VerticalScale":
                    params["vertical_scale"] = float(value)
                    found_params.add("VerticalScale")
                elif name == "VerticalOffset":
                    params["vertical_offset"] = float(value)
                    found_params.add("VerticalOffset")
                elif name == "ByteOrder":
                    if not value:
                        logger.warning(
                            f"Empty ByteOrder value in {sidecar_path}, using default LSB"
                        )
                        continue
                    params["byte_order"] = "LSB" if "LSB" in value else "MSB"
                    found_params.add("ByteOrder")
                elif name == "SignalFormat":
                    if not value:
                        logger.warning(
                            f"Empty SignalFormat value in {sidecar_path}, using default float32"
                        )
                        continue
                    if "FLOAT" in value:
                        params["signal_format"] = "float32"
                    elif "INT16" in value:
                        params["signal_format"] = "int16"
                    elif "INT32" in value:
                        params["signal_format"] = "int32"
                    else:
                        logger.warning(
                            f"Unknown SignalFormat '{value}' in {sidecar_path}, using default float32"
                        )
                    found_params.add("SignalFormat")
                elif name == "SignalHardwareRecordLength":
                    params["signal_hardware_record_length"] = int(value)
                    found_params.add("SignalHardwareRecordLength")
            except ValueError as e:
                logger.warning(
                    f"Failed to parse {name} value '{value}' in {sidecar_path}: {e}"
                )
                continue

        # Validate critical parameters
        if not found_resolution:
            warn(
                "Neither 'Resolution' nor 'SignalResolution' found in XML. "
                + "Using default sampling_interval=None. "
                + "Please provide a value or check your XML."
            )
        if (
            "SignalResolution" in found_params
            and "Resolution" in found_params
            and not np.isclose(signal_resolution, resolution, rtol=1e-2, atol=1e-9)
        ):
            logger.warning(
                f"FYI: 'Resolution' ({resolution}) != SignalResolution' ({signal_resolution}) found in {sidecar_path}. "
                f"Using 'Resolution' ({signal_resolution}). Diff: {abs(signal_resolution - resolution)}"
            )

        # Log what we found for debugging
        logger.debug(f"XML parsing found parameters: {found_params}")

        # Validate sampling interval if found
        if params["sampling_interval"] is not None and params["sampling_interval"] <= 0:
            logger.warning(
                f"Invalid sampling interval {params['sampling_interval']} in {sidecar_path}. "
                "This may lead to issues with time array generation."
            )

    except ET.ParseError as e:
        raise RuntimeError(f"XML parsing error in {sidecar_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse XML sidecar: {sidecar_path}: {e}")
    return params


def rd(
    filename: str,
    sampling_interval: Optional[float] = None,
    data_path: Optional[str] = None,
    sidecar: Optional[str] = None,
    crop: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read waveform binary file using sidecar XML for parameters.

    Parameters
    ----------
    filename : str
        Name of the binary waveform file.
    sampling_interval : float, optional
        Sampling interval in seconds. If None, reads from XML sidecar.
    data_path : str, optional
        Path to the data directory.
    sidecar : str, optional
        Name of the XML sidecar file.
    crop : List[int], optional
        Crop indices [start, end]. If None, uses entire signal.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Time array (float32) and scaled signal array (float32).

    Raises
    ------
    RuntimeError
        If sampling interval cannot be determined.
    FileNotFoundError
        If the binary file is not found.
    """
    # Always join data_path and filename if filename is not absolute
    if data_path is not None and not os.path.isabs(filename):
        fp = os.path.join(data_path, filename)
    else:
        fp = filename
    params = get_waveform_params(
        os.path.basename(fp), data_path, sidecar=sidecar
    )
    # Use sampling_interval from XML if available, else argument, else raise error
    si = params["sampling_interval"]
    if si is None:
        if sampling_interval is not None:
            si = sampling_interval
        else:
            raise RuntimeError(
                f"Sampling interval could not be determined for file: {fp}. "
                + "Please provide it or ensure the XML sidecar is present."
            )
    # log info about what we're reading and the parameters
    rel_fp = os.path.relpath(fp, os.getcwd()) if os.path.isabs(fp) else fp
    logger.info(f"Reading binary file: {rel_fp}")
    if sidecar:
        sidecar_path = _get_xml_sidecar_path(os.path.basename(fp), data_path, sidecar)
        rel_sidecar = (
            os.path.relpath(sidecar_path, os.getcwd())
            if os.path.isabs(sidecar_path)
            else sidecar_path
        )
        logger.info(f"--Using sidecar XML: {rel_sidecar}")
    logger.info(f"--Sampling interval: {si}")
    logger.info(f"--Vertical scale: {params['vertical_scale']}")
    logger.info(f"--Vertical offset: {params['vertical_offset']}")
    logger.info(f"--Byte order: {params['byte_order']}")
    logger.info(f"--Signal format: {params['signal_format']}")
    # Determine dtype
    dtype = np.float32
    if params["signal_format"] == "int16":
        dtype = np.int16
    elif params["signal_format"] == "int32":
        dtype = np.int32
    # Determine byte order
    byteorder = "<" if params["byte_order"] == "LSB" else ">"
    try:
        with open(fp, "rb") as f:
            import struct
            # Read first two bytes into two 32-bit unsigned integers,
            header_bytes = f.read(8)
            elsize, record_length_from_header = struct.unpack('<II', header_bytes)
            logger.success(f"Bin header: data el. size: {elsize} (bytes)")
            logger.success(f"Bin header: length: {record_length_from_header} ({elsize}-byte nums)")
            params["record_length_from_header"] = record_length_from_header
        if params["signal_hardware_record_length"] != record_length_from_header:
            logger.warning(
                f"SignalHardwareRecordLength ({params['signal_hardware_record_length']}) "
                f"does not match header record length ({record_length_from_header}) in {rel_fp}. "
                "This may indicate a mismatch in expected data length."
            )

        # first 8 bytes are the header (equiv to 2 float32s)
        arr = np.fromfile(fp, dtype=byteorder + dtype().dtype.char, offset=8)
        
        # Validate expected data length if available
        expected_length = params["signal_hardware_record_length"]
        if expected_length is not None:
            if len(arr) != expected_length:
                # raise RuntimeError(
                logger.warning(
                    f"Data length mismatch in {rel_fp}: "
                    f"expected {expected_length} points from SignalHardwareRecordLength, "
                    f"but read {len(arr)} points from binary file"
                )
        
        if crop is not None:
            x = arr[crop[0] : crop[1]]
        else:
            x = arr
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The file '{fp}' was not found. "
            + "Please ensure the file is in the correct directory."
        )
    # Scale and offset, ensuring float32 throughout
    scale = params["vertical_scale"] if params["vertical_scale"] is not None else 1.0
    offset = params["vertical_offset"] if params["vertical_offset"] is not None else 0.0
    x = (x * np.float32(scale) + np.float32(offset)).astype(np.float32)

    # Use np.linspace for more robust time array generation
    num_points = len(x)
    if num_points > 0:
        t = np.linspace(0, (num_points - 1) * si, num_points, dtype=np.float32)
    else:
        t = np.array([], dtype=np.float32)
        logger.warning(
            f"Generated an empty time array for file {rel_fp}. "
            f"Length of signal: {len(x)}, sampling interval: {si}. "
            "This might indicate an issue with input data or sampling interval."
        )

    return t, x
