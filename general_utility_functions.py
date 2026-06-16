"""
Small console-printing helpers used by the DyNeMo scripts.

These simply wrap ``print`` with ANSI colour codes so that progress messages
are easier to read in the terminal. They degrade gracefully on terminals that
do not support ANSI (the codes are just ignored / printed).
"""

# ANSI colour codes
_RESET = "\033[0m"
_CYAN = "\033[96m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"


def _cprint(msg, color):
    print(f"{color}{msg}{_RESET}")


def cprint(*args, **kwargs):
    """Cyan print (general progress / info)."""
    _cprint(" ".join(str(a) for a in args), _CYAN)


def rprint(*args, **kwargs):
    """Red print (warnings / errors)."""
    _cprint(" ".join(str(a) for a in args), _RED)


def yprint(*args, **kwargs):
    """Yellow print (cautions / things to review)."""
    _cprint(" ".join(str(a) for a in args), _YELLOW)


def gprint(*args, **kwargs):
    """Green print (success / OK)."""
    _cprint(" ".join(str(a) for a in args), _GREEN)

