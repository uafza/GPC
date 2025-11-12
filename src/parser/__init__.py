# src/parser/__init__.py
"""
SwissCAT+ parser package
Provides tools for reading, processing, and calibrating HPLC/GPC data.
"""

from .batch_parser import HPLCBatchParser
from .calibration_parser import GPCCalibrationParser
from .metadata_parser import MetadataParser
from .utils import construct_filename
