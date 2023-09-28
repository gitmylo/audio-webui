import sys

from .commands import run_command, get_python
from .os import is_windows
import os

venv_name = 'venv'
venv_activate_path = f'{venv_name}/' + ('Scripts/activate.bat' if is_windows() else 'bin/activate')


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix


def in_venv():
    in_conda = os.environ.get('CONDA_PREFIX') is not None
    return (get_base_prefix_compat() != sys.prefix) or in_conda



def activate_venv():
    if in_venv():
        return
    if not os.path.isdir(venv_name):
        print('no venv found, creating venv')
        run_command(f'"{get_python()}"', '-m venv venv')
    run_command([('call' if is_windows() else 'source', venv_activate_path), ('python', ' '.join([f'"{arg}"' for arg in sys.argv]))])  # Launch the main.py with the venv
    exit()  # Exit after the venv'ed version exits (maximum depth will be 2 because the venv is already activated in that case)


def ensure_venv():
    if not in_venv():
        print('activating venv')
        activate_venv()

