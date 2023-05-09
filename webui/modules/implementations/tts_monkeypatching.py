def patch():
    print('Monkeypatching bark')
    import bark
    import webui.modules.implementations.patches.bark_api as new_bark_api
    bark.api.generate_audio = new_bark_api.generate_audio_new
