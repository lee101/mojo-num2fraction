import json
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-num2fraction.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-num2fraction.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def upstream_vectors():
    """Strings produced by the real npm num2fraction, run under node.

    Regenerate with ``node tests/upstream_vectors.js <path to index.js>``.
    """
    return json.loads((_ROOT / "tests" / "upstream_vectors.json").read_text())


@pytest.fixture(scope="session")
def mmap_high():
    """A float64 array whose address is above 4 GiB.

    A ctypes ``argtypes`` list one entry short is silent: ctypes zero-fills the
    missing slot, a 64-bit address lands in a 32-bit one and is truncated.
    Every small-array test then passes while the shim reads or writes the low
    32 bits of the address -- the brk heap, which is mapped and writable, so it
    corrupts memory instead of crashing.  Allocations of a few MiB or more come
    back from mmap on the high half of the address space, which is what makes
    the truncation visible; a 64 MiB request reliably does.
    """
    a = np.zeros(8 * 1024 * 1024, dtype=np.float64)  # 64 MiB
    if a.ctypes.data < (1 << 32):
        pytest.skip("allocator returned a low address; cannot test truncation")
    return a
