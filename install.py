from webui.args import args


def ensure_installed():
    import webui.extensionlib.extensionmanager as em
    em.init_extensions()

    from setup_tools.magicinstaller.magicinstaller import install_requirements
    from setup_tools.venv import ensure_venv

    if not args.skip_venv:
        ensure_venv()
    if not args.skip_install:
        install_requirements()
    if args.download_models:
        from webui.modules import model_manager
        model_manager.download_all_models()


if __name__ == '__main__':
    ensure_installed()
