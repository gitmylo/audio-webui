from setup_tools.requirements_parser import install_requirements
from setup_tools.venv import ensure_venv


def ensure_installed():
    ensure_venv()
    install_requirements()
