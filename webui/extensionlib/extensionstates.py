import shlex
import subprocess

from setup_tools import os


def git_ready():
    cmd = 'git -v'
    cmd = cmd if os.is_windows() else shlex.split(cmd)
    result = subprocess.run(cmd).returncode
    return result == 0


class ExtensionState:
    def __init__(self, extension_dir_name):
        pass  # Load extension


states: dict[str, ExtensionState] = {}
