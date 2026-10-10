"""Single import point of validate_recovery(): the file that defines it starts with a digit and is not
a valid Python identifier, so it is loaded by path only once.
"""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def load_validate_recovery():
    """Loads and returns the validate_recovery() function of 4a_. Cached at module level (Python
    import) -- calling it several times does not reload the file.
    """
    spec = importlib.util.spec_from_file_location(
        "recovery_function", _ROOT / "model" / "4_recovery" / "4a_recovery_function.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate_recovery


validate_recovery = load_validate_recovery()
