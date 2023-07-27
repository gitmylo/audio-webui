from setup_tools.magicinstaller.requirement import SimpleRequirementInit


class NoColabRequirement(SimpleRequirementInit):
    def install(self) -> tuple[int, str, str]:
        try:
            import google.colab
            return 0, "", ""
        except:
            pass
        return super().install()
