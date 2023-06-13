import gradio
import webui.modules.implementations.audiocraft as acrft
from audiocraft.data.audio import audio_write


def generate(prompt, input_audio, top_k, top_p, temp, duration):
    output = acrft.generate(prompt, input_audio, True, top_k, top_p, temp, duration)
    if isinstance(output, str):
        return None, None, output
    else:
        return output, gradio.make_waveform(output)


def audiocraft_tab():
    with gradio.Row():
        with gradio.Column():
            with gradio.Row():
                selected = gradio.Dropdown(acrft.models, value='medium', label='Model')
                with gradio.Column(elem_classes='smallsplit'):
                    load = gradio.Button('🚀', variant='tool secondary')
                    unload = gradio.Button('💣', variant='tool primary')

                def load_model(model):
                    acrft.create_model(model)
                    return model

                def unload_model():
                    acrft.delete_model()
                    return ''

                load.click(load_model, selected, selected)
                unload.click(unload_model, outputs=selected)
            prompt = gradio.TextArea(label='Prompt', info='Put the audio you want here.', placeholder='Something like: "happy rock", "energetic EDM" or "sad jazz"\nLonger descriptions are also supported.')
            duration = gradio.Number(5, label='Duration (s)', info='Duration for the generation in seconds.')
            input_audio = gradio.Audio(label='Input audio for melody model')
            with gradio.Row():
                top_k = gradio.Slider(label='top_k', info='Higher number = more possible tokens, 0 to disable', minimum=0, value=250, maximum=10000, step=1)
                top_p = gradio.Slider(label='top_p', info='Higher number = more possible tokens, 0 to use top_k instead', minimum=0, value=0, maximum=1, step=0.01)
            temp = gradio.Slider(label='temperature', info='Higher number = more randomness for picking the next token', minimum=0, value=1, maximum=2)

        with gradio.Column():
            gen_button = gradio.Button('Generate', variant='primary')
            audio_out = gradio.Audio(label='Generated audio')
            video_out = gradio.Video(label='Waveform video')

        gen_button.click(generate, inputs=[prompt, input_audio, top_k, top_p, temp, duration], outputs=[audio_out, video_out])
