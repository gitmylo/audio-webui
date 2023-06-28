def install_requirements():
    from setup_tools.magicinstaller.requirements import requirements
    for requirement in requirements:
        requirement.install_or_upgrade_if_needed()
    print('Done installing/checking installs.')
