from setup_tools.magicinstaller.requirement import Requirement, SimpleRequirement


class Bark(SimpleRequirement):

    def is_installed(self):
        return self.install_check('suno-bark')

    def install(self) -> tuple[int, str, str]:
        return self.install_pip('git+https://github.com/suno-ai/bark.git@6921c9139a97d0364208407191c92ec265ef6759', 'bark')


class SoundFileOrSox(SimpleRequirement):

    def is_installed(self):
        if self.is_windows():
            return self.install_check('soundfile')
        else:
            return self.install_check('sox')

    def install(self) -> tuple[int, str, str]:
        if self.is_windows():
            return self.install_pip('soundfile')
        else:
            return self.install_pip('sox')
