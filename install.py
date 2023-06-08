from setup_tools.requirements_parser import install_requirements
from setup_tools.venv import ensure_venv
from webui.args import args


def ensure_installed():
    if not args.skip_venv:
        ensure_venv()
    if not args.skip_install:
        install_requirements(show_output=not args.hide_pip_log)


if __name__ == '__main__':
    ensure_installed()
