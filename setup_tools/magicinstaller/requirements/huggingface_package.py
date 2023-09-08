from setup_tools.magicinstaller.requirement import SimpleRequirement, SimpleGitRequirement


class Transformers(SimpleRequirement):
    package_name = 'transformers'


def diffusers():
    return SimpleGitRequirement('diffusers', 'git+https://github.com/huggingface/diffusers@d73e6ad050ee4831bc367d45a3af9750e1204dae', True)
