from webui import args  # Will show help message if needed

from install import ensure_installed

print('Checking installs and venv')

ensure_installed()  # Installs missing packages

print('Launching')

from webui import launch_webui
launch_webui()
