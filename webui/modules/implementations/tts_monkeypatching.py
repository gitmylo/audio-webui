def patch():
    print('Monkeypatching bark')
    import bark
    import webui.modules.implementations.patches.bark_api as new_bark_api
    bark.api.generate_audio = new_bark_api.generate_audio_new
    print('Monkeypatching TTS')
    import TTS.api
    import TTS.vc.models
    import webui.modules.implementations.patches.frog as new_TTS
    TTS.api.TTS = new_TTS.TTS
