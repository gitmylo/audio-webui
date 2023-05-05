import sys
from .os import is_windows
import os

venv_name = 'venv'
venv_activate_path = f'{venv_name}/' + 'Scripts/activate.bat' if is_windows() else 'bin/activate'


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix


def in_venv():
    return get_base_prefix_compat() != sys.prefix


def activate_venv():
    if not os.path.isdir(venv_name):
        pass
    # Launch the main.py with the venv
    exit()  # Exit after the venv'ed version exits


def ensure_venv():
    if not in_venv():
        print('activating venv')
        activate_venv()
    else:
        print('in venv')
    print('after venv in venv:', in_venv())
