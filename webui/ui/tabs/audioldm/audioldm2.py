import gradio

import webui.modules.implementations.audioldm2 as aldm2
from webui.modules import util


def generate(prompt, negative, duration, steps, cfg, seed, progress=gradio.Progress()):
    output = aldm2.generate(prompt, negative, steps, duration, cfg, seed,
                           callback=lambda step, _, _2: progress((step, steps), desc='Generating...'))
    if isinstance(output, str):
        return None, None, output
    else:
        return output[1], util.make_waveform(output[1]), f'Successfully generated audio with seed: {output[0]}.'


def audioldm2_tab():
    with gradio.Row():
        with gradio.Row():
            selected = gradio.Dropdown(aldm2.models, value='cvssp/audioldm2',
                                       label='Model')
            with gradio.Column(elem_classes='smallsplit'):
                load_button = gradio.Button('🚀', variant='tool secondary')
                unload_button = gradio.Button('💣', variant='tool primary')

            def load(model):
                aldm2.create_model(model)
                return gradio.update()

            def unload():
                aldm2.delete_model()
                return gradio.update()

            load_button.click(fn=load, inputs=selected, outputs=selected, show_progress=True)
            unload_button.click(fn=unload, outputs=selected, show_progress=True)
        with gradio.Row():
            gen_button = gradio.Button('Generate', variant='primary')

    with gradio.Row():
        with gradio.Column():
            prompt = gradio.TextArea(label='Prompt', info='Put the audio you want here.',
                                     placeholder='The sound of a hammer hitting a wooden surface')
            neg_prompt = gradio.TextArea(label='Negative prompt', info='Put things to avoid generating here.',
                                         placeholder='low bitrate, low quality, bad quality')
            duration = gradio.Number(5, label='Duration (s)', info='Duration for the generation in seconds.')
            seed = gradio.Number(-1, label='Seed',
                                 info='Default: -1 (random). Set the seed for generation, random seed is used when a negative number is given.',
                                 precision=0)
            with gradio.Accordion('➕ Extra options', open=False):
                cfg = gradio.Slider(1, 20, 3.5, step=0.01, label='CFG scale',
                                    info='Default: 2.5. How much should the prompt affect the audio on every step?')
                steps = gradio.Slider(1, 300, 50, step=1, label='Steps',
                                      info='Default: 10. How many diffusion steps should be performed?')
        with gradio.Column():
            with gradio.Row():
                audio_out = gradio.Audio(label='Generated audio', interactive=False)
            with gradio.Row():
                video_out = gradio.Video(label='Waveform video', interactive=False)
            with gradio.Row():
                text_out = gradio.Textbox(label='Result')

    gen_button.click(fn=generate, inputs=[prompt, neg_prompt, duration, steps, cfg, seed],
                     outputs=[audio_out, video_out, text_out])
