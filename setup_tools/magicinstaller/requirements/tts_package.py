from setup_tools.magicinstaller.requirement import SimpleRequirement


class TTS(SimpleRequirement):
    package_name = 'TTS'

    def install(self) -> tuple[int, str, str]:
        return self.install_pip("tts[ja]", "TTS")
