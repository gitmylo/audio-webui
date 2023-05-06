import gradio
import webui.modules.models as mod
import webui.modules.implementations.ttsmodels as tts_models

mod_type = 'text-to-speech'

loader: mod.TTSModelLoader = mod.TTSModelLoader


def get_models_installed():
    return [model for model in mod.get_installed_models(mod_type) if model in [tts.replace('/', '--') for tts in mod.all_tts_models()]]


def text_to_speech():
    with gradio.Row():
        with gradio.Column():
            # input_box = gradio.Textbox(lines=7, label='Input', placeholder='Text to speak goes here')
            all_components_dict = tts_models.all_elements_dict()
            all_components = tts_models.all_elements(all_components_dict)
            with gradio.Row():
                selected = gradio.Dropdown(get_models_installed(), label='Model')
                with gradio.Column(elem_classes='smallsplit'):
                    refresh = gradio.Button('🔃', variant='tool secondary')
                    unload = gradio.Button('💣', variant='tool primary')
                refresh.click(fn=get_models_installed, outputs=selected, show_progress=True)

                def unload_model():
                    global loader
                    if isinstance(loader, mod.TTSModelLoader):
                        loader.unload_model()
                    return [gradio.update(value='')] + [gradio.update(visible=False) for e in all_components]
                unload.click(fn=unload_model, outputs=[selected] + all_components, show_progress=True)

                def load_model(model):
                    unload_model()
                    global loader
                    loader = loader.from_model(model)
                    inputs = all_components_dict[loader.model]
                    return_value = [gradio.update()] + [gradio.update(visible=element in inputs) for element in all_components]
                    return return_value
                selected.select(fn=load_model, inputs=selected, outputs=[selected] + all_components, show_progress=True)
        with gradio.Column():
            generate = gradio.Button('Generate')
            audio_out = gradio.Audio()

    def _generate(*inputs):
        global loader
        inputs = [inp for inp in inputs if inp in all_components_dict[loader.model]]  # Filter inputs
        return loader.get_response(*inputs)
    generate.click(fn=_generate, inputs=all_components, outputs=audio_out, show_progress=True)
