from setup_tools.magicinstaller.magicinstaller import install_requirements
from setup_tools.venv import ensure_venv
from webui.args import args


def ensure_installed():
    if not args.skip_venv:
        ensure_venv()
    install_requirements()


if __name__ == '__main__':
    ensure_installed()
