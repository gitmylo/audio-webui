import gradio
from gradio.components import IOComponent

import webui.modules.models as mod
import webui.modules.implementations.ttsmodels as tts_models

mod_type = 'text-to-speech'

loader: mod.TTSModelLoader = mod.TTSModelLoader

to_rvc, audio_out = None, None


def get_models_installed():
    # return [model for model in mod.get_installed_models(mod_type) if model in [tts.replace('/', '--') for tts in mod.all_tts_models()]]
    return [model for model in mod.get_installed_models(mod_type) if
            model in [tts.replace('/', '--') for tts in mod.all_tts_models()]] + \
           [model.model for model in mod.all_tts() if model.no_install]


def filter_components(components):
    return [component for component in components if isinstance(component, IOComponent)]


def text_to_speech():
    with gradio.Row():
        with gradio.Column():
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
                    return [gradio.update(value='')] + [gradio.update(visible=False) for _ in all_components]

                unload.click(fn=unload_model, outputs=[selected] + all_components, show_progress=True)

                def load_model(model):
                    global loader
                    if not (hasattr(loader, 'model') and model.lower().endswith(loader.model.lower())):
                        unload_model()
                    loader = loader.from_model(model)
                    loader.load_model()
                    inputs = all_components_dict[loader.model]
                    return_value = [gradio.update()] + [
                        gradio.update(visible=element in inputs and not (hasattr(element, 'hide') and element.hide)) for
                        element in all_components]
                    return return_value

                selected.select(fn=load_model, inputs=selected, outputs=[selected] + all_components, show_progress=True)
        with gradio.Column():
            global to_rvc, audio_out
            with gradio.Row():
                generate = gradio.Button('Generate', variant='primary')
                to_rvc = gradio.Button('Send to RVC')
            audio_out = gradio.Audio(interactive=False)
            video_out = gradio.Video()
            file_out = gradio.File()



    def _generate(inputs, values):
        global loader
        inputs = [values[i] for i in range(len(inputs)) if
                  inputs[i] in all_components_dict[loader.model]]  # Filter and convert inputs
        response, file = loader.get_response(*inputs)
        return response, gradio.make_waveform(response), file

    filtered_components = filter_components(all_components)
    generate.click(fn=lambda *values: _generate(filtered_components, values), inputs=filtered_components,
                   outputs=[audio_out, video_out, file_out], show_progress=True)
