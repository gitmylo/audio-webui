from logging import info
from install import ensure_installed

info('Checking installs and venv')

ensure_installed()  # Installs missing packages
