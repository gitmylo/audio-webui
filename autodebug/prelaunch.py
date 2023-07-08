import shlex
import subprocess

from install import ensure_installed
import sys
import autodebug.autodebug as autodebug
from setup_tools import os


def check_python():
    print('Python version: ', sys.version)
    major, minor, patch, variant, _ = sys.version_info
    if major == 3 and minor == 10:
        return
    raise autodebug.WrongPythonVersionException(f'Your python version is not supported. You\'re running "{major}.{minor}.{patch}". But you need "3.10.x"')


def print_git():
    command = 'git log --pretty="Webui version: %H - %cd" -n 1'
    command = command if os.is_windows() else shlex.split(command)
    result = subprocess.run(command, capture_output=True)
    if result.returncode == 0:
        print(result.stdout.decode(encoding=sys.getdefaultencoding()), end='')
    else:
        print('Webui version: Unable to check version, not installed with git.')


def prelaunch_checks():
    check_python()
    print_git()

    ensure_installed()  # Installs missing packages
