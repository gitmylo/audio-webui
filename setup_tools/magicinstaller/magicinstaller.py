def install_requirements():
    from setup_tools.magicinstaller.requirements import requirements
    import webui.extensionlib.extensionmanager as em
    for requirement in requirements + em.get_requirements():
        requirement.install_or_upgrade_if_needed()

    import importlib
    importlib.invalidate_caches()

    print('Done installing/checking installs.')
