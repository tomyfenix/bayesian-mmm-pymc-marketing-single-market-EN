"""HOW A FIT IS COMPILED ON THIS MACHINE -- one single place.
"""

from contextlib import contextmanager

NUMBA_KWARGS = {"compile_kwargs": {"mode": "NUMBA"}}


@contextmanager
def all_in_numba():
    """While the block lasts, whatever is compiled without an explicit mode is compiled in NUMBA. On
    exit the previous mode comes back (change_flags restores it even if the fit fails).
    """
    import pytensor
    with pytensor.config.change_flags(mode="NUMBA"):
        yield
