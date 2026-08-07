import os
import sys

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_BASE_DIR = os.path.join(_SRC_DIR, "synthetic_fed_hnet_lora")
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
