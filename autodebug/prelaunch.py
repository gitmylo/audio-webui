from install import ensure_installed
import sys
import autodebug.autodebug as autodebug


def check_python():
    print('Python version: ', sys.version)
    major, minor, patch, variant, _ = sys.version_info
    if major == 3 and minor == 10:
        return
    raise autodebug.WrongPythonVersionException(f'Your python version is not supported. You\'re running "{major}.{minor}.{patch}". But you need "3.10.x"')


def prelaunch_checks():
    check_python()

    ensure_installed()  # Installs missing packages
