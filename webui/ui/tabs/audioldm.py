import gradio
import webui.modules.implementations.audioldm as aldm
from webui.modules import util


def generate(prompt, negative, duration, steps, cfg, seed, wav_best_count, enhance, progress=gradio.Progress()):
    output = aldm.generate(prompt, negative, steps, duration, cfg, seed, wav_best_count, enhance,
                           callback=lambda step, _, _2: progress((step, steps), desc='Generating...'))
    if isinstance(output, str):
        return None, None, output
    else:
        return output[1], util.make_waveform(output[1]), f'Successfully generated audio with seed: {output[0]}.'


def audioldm_tab():
    with gradio.Row():
        with gradio.Row():
            selected = gradio.Dropdown(aldm.models, value='cvssp/audioldm-m-full', label='Model (in order of small to large, old to new)')
            with gradio.Column(elem_classes='smallsplit'):
                load_button = gradio.Button('🚀', variant='tool secondary')
                unload_button = gradio.Button('💣', variant='tool primary')

            def load(model):
                aldm.create_model(model)
                return gradio.update()

            def unload():
                aldm.delete_model()
                return gradio.update()

            load_button.click(fn=load, inputs=selected, outputs=selected, show_progress=True)
            unload_button.click(fn=unload, outputs=selected, show_progress=True)
        with gradio.Row():
            gen_button = gradio.Button('Generate', variant='primary')

    with gradio.Row():
        with gradio.Column():
            prompt = gradio.TextArea(label='Prompt', info='Put the audio you want here.',
                                     placeholder='Techno music with a strong, upbeat tempo and high melodic riffs')
            neg_prompt = gradio.TextArea(label='Negative prompt', info='Put things to avoid generating here.',
                                         placeholder='low bitrate, low quality, bad quality')
            duration = gradio.Number(5, label='Duration (s)', info='Duration for the generation in seconds.')
            seed = gradio.Number(-1, label='Seed',
                                 info='Default: -1 (random). Set the seed for generation, random seed is used when a negative number is given.', precision=0)
            with gradio.Accordion('➕ Extra options', open=False):
                cfg = gradio.Slider(1, 20, 2.5, step=0.01, label='CFG scale',
                                    info='Default: 2.5. How much should the prompt affect the audio on every step?')
                steps = gradio.Slider(1, 150, 10, step=1, label='Steps',
                                      info='Default: 10. How many diffusion steps should be performed?')
                wav_best_count = gradio.Slider(0, 25, 3, step=1, label='Count',
                                               info='Default: 3. How generations to do? Best result will be picked.')
                enhance = gradio.Checkbox(label='Enhance output', info='This could sound better, but could also sound worse.', value=True)
        with gradio.Column():
            with gradio.Row():
                audio_out = gradio.Audio(label='Generated audio', interactive=False)
            with gradio.Row():
                video_out = gradio.Video(label='Waveform video', interactive=False)
            with gradio.Row():
                text_out = gradio.Textbox(label='Result')

    gen_button.click(fn=generate, inputs=[prompt, neg_prompt, duration, steps, cfg, seed, wav_best_count, enhance],
                     outputs=[audio_out, video_out, text_out])
