import shlex
import subprocess

from setup_tools import os


def git_ready():
    cmd = 'git -v'
    cmd = cmd if os.is_windows() else shlex.split(cmd)
    result = subprocess.run(cmd).returncode
    return result == 0


class ExtensionInfo:
    def __init__(self, name=None, description=None, authors=None):
        if not authors:
            authors = []

        self.name = name
        self.description = description,
        self.authors = authors

