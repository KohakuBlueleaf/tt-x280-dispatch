"""Test setup: offline tests import tt_x280 without ttnn or the _ttx280 module.

Hardware tests (marker `hw`) run only with TT_X280_HW=1 and the real modules present.
"""

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "frontend"))

# Stand-ins for the device modules when they are not installed.
for name in ("ttnn", "_ttx280"):
    try:
        __import__(name)
    except ImportError:
        module = types.ModuleType(name)
        module.__dict__.update(
            CTL_BYTES=1024,
            CTL_CB=8,
            TRACE_CB=10,
            IMAGE_DIR="",
            KERNEL_DIR="",
            MTIME_HZ=50e6,
            Plan=object,
            Runtime=object,
            Session=object,
        )
        sys.modules[name] = module


def pytest_collection_modifyitems(config, items):
    """Skip `hw` tests unless TT_X280_HW=1."""
    if os.environ.get("TT_X280_HW") == "1":
        return
    skip = pytest.mark.skip(reason="needs a p150a with a primed X280 (TT_X280_HW=1)")
    for item in items:
        if "hw" in item.keywords:
            item.add_marker(skip)
