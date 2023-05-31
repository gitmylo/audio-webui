from TTS.api import TTS
import gradio

flag_strings = ['denoise', 'separate music']


def get_models_installed():
    return []


def unload_rvc():
    return gradio.update()


def rvc():
    all_tts = TTS.list_models()
    with gradio.Row():
        with gradio.Column():
            with gradio.Accordion('TTS', open=False):
                selected_tts = gradio.Dropdown(all_tts, label='TTS model', info='The TTS model to use for text-to-speech')
                text_input = gradio.TextArea(label='Text to speech text', info='Text to speech text if no audio file is used as input.')
            with gradio.Accordion('Audio input', open=False):
                audio_input = gradio.Audio(label='Audio input')
            with gradio.Accordion('RVC'):
                with gradio.Row():
                    selected = gradio.Dropdown(get_models_installed(), label='RVC Model')
                    with gradio.Column(elem_classes='smallsplit'):
                        refresh = gradio.Button('🔃', variant='tool secondary')
                        unload = gradio.Button('💣', variant='tool primary')
                    refresh.click(fn=get_models_installed, outputs=selected, show_progress=True)
                    unload.click(fn=unload_rvc, outputs=selected, show_progress=True)
            flags = gradio.Dropdown(flag_strings, label='Flags', info='Things to apply on the audio input/output', multiselect=True)
        with gradio.Column():
            generate = gradio.Button('Generate')
            audio_out = gradio.Audio()
            video_out = gradio.Video()

        def gen(tts, text_in, audio_in, flag):
            pass
        generate.click(fn=gen, inputs=[selected_tts, text_input, audio_input, flags], outputs=[audio_out, video_out])
