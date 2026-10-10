"""Ask the user, or fail with a clear message if nobody is watching. A script that needs a decision
never makes it on its own.
"""

import sys


def _stdin_is_console() -> bool:
    """Standard input is a CONSOLE where someone can type. isatty() alone is NOT enough on Windows: it
    is True for NUL, the null device, too, so `script < NUL` (or `< /dev/null` from Git Bash) looks
    like a console and is not one.
    """
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        import msvcrt
        mode = ctypes.c_ulong()
        h = msvcrt.get_osfhandle(sys.stdin.fileno())
        return bool(ctypes.windll.kernel32.GetConsoleMode(h, ctypes.byref(mode)))
    except Exception:  # noqa: BLE001 -- when in doubt, there is no console
        return False


def someone_watching() -> bool:
    """True only if someone can answer now. When in doubt, False: asking nobody leaves a script hanging
    all night, and failing with a message breaks nothing.
    """
    if not _stdin_is_console():
        return False
    return True


class MissingDecision(SystemExit):
    """The script needs a decision that changes the result and there is nobody to ask for it."""


def missing_decision(what: str, how_to_pass_it: str):
    raise MissingDecision(
        f"\nA DECISION IS MISSING: {what}\n"
        f"There is nobody watching to ask, and this script does not choose on its own.\n"
        f"Pass it explicitly:  {how_to_pass_it}\n")


def _flag_value(flag: str, argv: list[str]) -> str | None:
    for i, a in enumerate(argv):
        if a == flag:
            if i + 1 >= len(argv):
                raise MissingDecision(f"\n{flag} needs a value.\n")
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def choose(question: str, options: dict[str, str], *, flag: str, argv: list[str],
           how_to_pass_it: str | None = None) -> str:
    """An option that changes the result: explicit by argument, or asked, or it fails. If `flag value`
    comes in argv, it is used -- and if the value is not an option, it fails (it is not "corrected" to
    the closest one).
    """
    keys = list(options)
    how = how_to_pass_it or f"{flag} {'|'.join(keys)}"
    value = _flag_value(flag, argv)
    if value is not None:
        if value not in options:
            raise MissingDecision(f"\n{flag} {value!r} is not a valid option. Options: {keys}\n")
        return value
    if not someone_watching():
        missing_decision(question, how)
    print("\n" + question)
    for i, k in enumerate(keys, 1):
        print(f"  {i}) {k:10} {options[k]}")
    while True:
        try:
            r = input(f"Choose 1-{len(keys)} (or type the name): ").strip()
        except EOFError:
            missing_decision(question, how)
        if r.isdigit() and 1 <= int(r) <= len(keys):
            return keys[int(r) - 1]
        if r in options:
            return r
        print(f"  {r!r} is not an option. There is no default: one has to be chosen.")


def flag_value(flag: str, argv: list[str]) -> str | None:
    """The value of `flag` in argv (`--x v` or `--x=v`), or None. Public: config/dgp_axes.py uses it to
    read every axis before asking anything.
    """
    return _flag_value(flag, argv)


def ask_text(question: str, *, validate, flag: str, argv: list[str], how_to_pass_it: str,
             help_text: str = "") -> str:
    """Like choose(), for a FREE value (a seed, a list): explicit by argument, or asked, or it fails.
    """
    value = _flag_value(flag, argv)
    if value is not None:
        try:
            validate(value)
        except ValueError as exc:
            raise MissingDecision(f"\n{flag} {value!r}: {exc}\n") from None
        return value
    if not someone_watching():
        missing_decision(question, how_to_pass_it)
    print("\n" + question)
    if help_text:
        print(help_text)
    while True:
        try:
            r = input("> ").strip()
        except EOFError:
            missing_decision(question, how_to_pass_it)
        if not r:
            print("  There is no default: a value has to be typed.")
            continue
        try:
            validate(r)
            return r
        except ValueError as exc:
            print(f"  {r!r}: {exc}")
