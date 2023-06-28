from setup_tools.magicinstaller.requirement import SimpleRequirement


class AudioCraft(SimpleRequirement):
    package_name = 'audiocraft'

    def install(self) -> tuple[int, str, str]:
        return self.install_pip('git+https://github.com/facebookresearch/audiocraft.git', 'audiocraft')
