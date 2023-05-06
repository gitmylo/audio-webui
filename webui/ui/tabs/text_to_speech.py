import gradio
import webui.modules.models as mod

mod_type = 'text-to-speech'


def text_to_speech():
    loader = mod.TTSModelLoader
    with gradio.Row():
        with gradio.Column():
            input_box = gradio.Textbox(lines=7, label='Input', placeholder='Text to speak goes here')
            with gradio.Row():
                selected = gradio.Dropdown(mod.get_installed_models(mod_type), label='Model')
                with gradio.Column(elem_classes='smallsplit'):
                    refresh = gradio.Button('🔃', variant='tool secondary')
                    load = gradio.Button('💣', variant='tool primary')
                refresh.click(fn=lambda: mod.get_installed_models(mod_type), outputs=selected, show_progress=True)
                load.click(fn=loader.from_model, inputs=selected, outputs=selected, show_progress=True)
        with gradio.Column():
            generate = gradio.Button('Generate')
            audio_out = gradio.Audio()

    generate.click(fn=None, inputs=input_box, outputs=audio_out)
