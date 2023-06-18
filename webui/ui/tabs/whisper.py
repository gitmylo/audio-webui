import gradio
import webui.modules.implementations.whisper as w


def whisper():
    with gradio.Row():
        with gradio.Row():
            selected = gradio.Dropdown(w.get_official_models(), value='base', label='Model')
            with gradio.Column(elem_classes='smallsplit'):
                load = gradio.Button('🚀', variant='tool secondary')
                unload = gradio.Button('💣', variant='tool primary')

            def load_model(model):
                return w.load(model)
        with gradio.Row():
            transcribe = gradio.Button('Transcribe', variant='primary')
    with gradio.Row():
        audio = gradio.Audio(label='Audio to transcribe')
        output = gradio.TextArea(label='Transcript')

    unload.click(fn=w.unload, outputs=output, show_progress=True)
    load.click(fn=load_model, inputs=selected, outputs=output, show_progress=True)

    transcribe.click(fn=w.transcribe, inputs=audio, outputs=output)
