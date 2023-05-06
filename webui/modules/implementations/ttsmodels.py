import webui.modules.models as mod


class BarkTTS(mod.TTSModelLoader):
    model = 'suno/bark'

    def get_response(self, *inputs):
        pass

    def unload_model(self, name):
        pass

    def load_model(self):
        pass


def all_tts() -> list[mod.TTSModelLoader]:
    return [BarkTTS]
