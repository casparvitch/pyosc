"""
Waveform I/O utilities.
"""

from .io import _get_xml_sidecar_path, get_waveform_params, rd, rd_chunked

__all__ = ["rd", "rd_chunked", "get_waveform_params", "_get_xml_sidecar_path"]
