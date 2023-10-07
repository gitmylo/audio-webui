import gc
import os.path
import tempfile

import gradio
import numpy
import numpy as np
import requests
import scipy.io.wavfile
import torch.cuda
# from TTS.api import TTS
# from TTS.utils.manage import ModelManager

import webui.modules.models as mod
from webui.modules.implementations.patches.bark_custom_voices import wav_to_semantics, generate_fine_from_wav, \
    generate_course_history
from webui.ui.tabs import settings

hubert_models_cache = None


class BarkTTS(mod.TTSModelLoader):
    no_install = True

    @staticmethod
    def get_voices():
        found_prompts = []
        base_path = 'data/bark_custom_speakers/'
        for path, subdirs, files in os.walk(base_path):
            for name in files:
                if name.endswith('.npz'):
                    found_prompts.append(os.path.join(path, name)[len(base_path):-4])
        from webui.modules.implementations.patches.bark_generation import ALLOWED_PROMPTS
        return ['None'] + found_prompts + ALLOWED_PROMPTS

    @staticmethod
    def create_voice(file, clone_model):
        clone_model_obj = [model for model in hubert_models_cache if model['name'].casefold() == clone_model.casefold()][0]
        file_name = '.'.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
        out_file = f'data/bark_custom_speakers/{file_name}.npz'

        semantic_prompt = wav_to_semantics(file, clone_model_obj)
        fine_prompt = generate_fine_from_wav(file)
        coarse_prompt = generate_course_history(fine_prompt)


        np.savez(out_file,
                 semantic_prompt=semantic_prompt,
                 fine_prompt=fine_prompt,
                 coarse_prompt=coarse_prompt
                 )
        return file_name

    @staticmethod
    def get_cloning_models():
        global hubert_models_cache
        if hubert_models_cache:
            return hubert_models_cache
        try:
            r = requests.get('https://raw.githubusercontent.com/gitmylo/Voice-cloning-quantizers/main/models.json')
            hubert_models_cache = r.json()
        except:  # No internet connection or something similar
            hubert_models_cache = [
                {
                    "name": "Base English",
                    "repo": "GitMylo/bark-voice-cloning",
                    "file": "quantifier_hubert_base_ls960_14.pth",
                    "language": "ENG",
                    "author": "https://github.com/gitmylo/",
                    "quant_version": 0,
                    "official": True,
                    "dlfilename": "tokenizer.pth",
                    "extra": {
                        "dataset": "https://huggingface.co/datasets/GitMylo/bark-semantic-training"
                    }
                },
                {
                    "name": "Large English",
                    "repo": "GitMylo/bark-voice-cloning",
                    "file": "quantifier_V1_hubert_base_ls960_23.pth",
                    "language": "ENG",
                    "author": "https://github.com/gitmylo/",
                    "quant_version": 1,
                    "official": True,
                    "dlfilename": "tokenizer_large.pth",
                    "extra": {
                        "dataset": "https://huggingface.co/datasets/GitMylo/bark-semantic-training"
                    }
                }
            ]
        return hubert_models_cache

    def _components(self, **quick_kwargs):
        def update_speaker(option):
            if option == 'File':
                speaker.hide = False
                refresh_speakers.hide = False
                speaker_file.hide = True
                speaker_name.hide = True
                clone_model.hide = True
                npz_file.hide = True
                speakers.hide = False
                return [gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False)]
            elif option == 'Clone':
                speaker.hide = True
                refresh_speakers.hide = True
                speaker_file.hide = True
                speaker_name.hide = False
                clone_model.hide = False
                npz_file.hide = True
                speakers.hide = True
                return [gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=False)]
            elif option == 'Upload .npz':
                speaker.hide = True
                refresh_speakers.hide = True
                speaker_file.hide = True
                speaker_name.hide = True
                clone_model.hide = True
                npz_file.hide = False
                speakers.hide = True
                return [gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=False),
                        gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True)]

        def update_input(option):
            if option == 'Text':
                textbox.hide = False
                split_type.hide = False
                audio_upload.hide = True
                return [gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True)]
            else:
                textbox.hide = True
                split_type.hide = True
                audio_upload.hide = False
                return [gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=False)]

        def update_voices():
            return gradio.update(choices=self.get_voices())

        clone_models = [m['name'] for m in self.get_cloning_models()]

        input_type = gradio.Radio(['Text', 'Audio'], label='Input type', value='Text', **quick_kwargs)
        textbox = gradio.Textbox(lines=7, label='Input', placeholder='Text to speak goes here', info='For manual splitting, use enter. Otherwise, don\'t worry about it', **quick_kwargs)
        split_type = gradio.Dropdown(['Manual', 'Strict short', 'Strict long', 'Non-strict short', 'Non-strict long'], value='Strict long', label='Splitting type', **quick_kwargs)

        gen_prefix = gradio.Textbox(label='Generation prefix', info='Add this text before every generated chunk, better for keeping emotions.', **quick_kwargs)
        input_lang_model = gradio.Dropdown(clone_models, value=clone_models[0], label='Speech recognition bark quantizer.', info='The "voice cloning" model to use. Mainly for languages.', **quick_kwargs)
        audio_upload = gradio.File(label='Words to speak', file_types=['audio'], **quick_kwargs)
        input_lang_model.hide = True
        audio_upload.hide = True
        # with gradio.Row(visible=False) as temps:
        text_temp = gradio.Slider(0.05, 1.5, 0.7, step=0.05, label='Text temperature', info='Affects the randomness of the generated speech patterns, like with Language models, higher is more random', **quick_kwargs)
        waveform_temp = gradio.Slider(0.05, 1.5, 0.7, step=0.05, label='Waveform temperature', info='Affects the randomness of the audio generated from the previous generated speech patterns, like with Language models, higher is more random', **quick_kwargs)

        with gradio.Accordion(label='Voice cloning guide and long form generations', open=False, visible=False) as a:
            clone_guide = gradio.Markdown('''
            ## Long form generations
            Split your long form generations with newlines (enter), every line will be generated individually, but as a continuation of the last.

            Empty lines at the start and end will be skipped.

            ## When cloning a voice:
            * The speaker will be saved in the data/bark_custom_speakers directory.
            * The "file" output contains a different speaker. This is for saving speakers created through random generation. Or continued cloning.

            ## Cloning guide (short edition)
            * Clear spoken, no noise, no music.
            * Ends after a short pause for best results.
            * The speaker will be saved in the data/bark_custom_speakers directory.
            * The “file” output contains a different speaker. This is for saving speakers created through random generation. Or continued cloning.
                    ''', visible=False)

        mode = gradio.Radio(['File', 'Clone', 'Upload .npz'], label='Speaker from', value='File', **quick_kwargs)
        with gradio.Row(visible=False) as speakers:
            speaker = gradio.Dropdown(self.get_voices(), value='None', show_label=False, **quick_kwargs)
            refresh_speakers = gradio.Button('🔃', variant='tool secondary', **quick_kwargs)
        refresh_speakers.click(fn=update_voices, outputs=speaker)
        clone_model = gradio.Dropdown(clone_models, value=clone_models[0], label='Voice cloning model.', info='The voice cloning model to use. Mainly for languages.', **quick_kwargs)
        speaker_name = gradio.Textbox(label='Speaker name', info='The name to save the speaker as, random if empty', **quick_kwargs)
        speaker_file = gradio.Audio(label='Speaker', **quick_kwargs)
        clone_model.hide = True
        speaker_name.hide = True
        speaker_file.hide = True  # Custom, auto hide speaker_file

        npz_file = gradio.File(label='Npz file', file_types=['.npz'], **quick_kwargs)
        npz_file.hide = True

        keep_generating = gradio.Checkbox(label='Keep it up (keep generating)', value=False, **quick_kwargs)
        min_eos_p = gradio.Slider(0.05, 1, 0.2, step=0.05, label='min end of audio probability', info='Lower values cause the generation to stop sooner, higher values make it do more, 1 is about the same as keep generating being on.', **quick_kwargs)

        mode.change(fn=update_speaker, inputs=mode, outputs=[speakers, speaker, refresh_speakers, speaker_file, speaker_name, clone_model, npz_file])
        input_type.change(fn=update_input, inputs=input_type, outputs=[textbox, split_type, audio_upload, input_lang_model, gen_prefix])
        return [textbox, gen_prefix, audio_upload, input_type, mode, text_temp, waveform_temp,
                speaker, speaker_name, speaker_file, refresh_speakers, keep_generating, clone_guide, speakers, min_eos_p, a, clone_model, input_lang_model,
                npz_file, split_type]

    model = 'suno/bark'

    def get_response(self, *inputs, progress=gradio.Progress()):
        textbox, gen_prefix, audio_upload, input_type, mode, text_temp, waveform_temp, speaker,\
            speaker_name, speaker_file, refresh_speakers, keep_generating, clone_guide, min_eos_p, clone_model,\
            input_lang_model, npz_file, split_type = inputs
        _speaker = None
        if mode == 'File':
            _speaker = speaker if speaker != 'None' else None
        elif mode == 'Clone':
            speaker_sr, speaker_wav = speaker_file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            if speaker_name:
                temp_file.name = os.path.join(os.path.dirname(temp_file.name), speaker_name + '.wav')
            scipy.io.wavfile.write(temp_file.name, speaker_sr, speaker_wav)
            _speaker = self.create_voice(temp_file.name, clone_model)
        elif mode == 'Upload .npz':
            _speaker = npz_file.name
        from webui.modules.implementations.patches.bark_api import generate_audio_new, semantic_to_waveform_new
        from bark.generation import SAMPLE_RATE
        if input_type == 'Text':
            history_prompt, audio = generate_audio_new(textbox, _speaker, text_temp, waveform_temp, output_full=True,
                                                       allow_early_stop=not keep_generating, min_eos_p=min_eos_p,
                                                       gen_prefix=gen_prefix, progress=progress, split_type=split_type)
        else:
            input_lang_model_obj = \
            [model for model in hubert_models_cache if model['name'].casefold() == input_lang_model.casefold()][0]
            semantics = wav_to_semantics(audio_upload.name, input_lang_model_obj).numpy()
            history_prompt, audio = semantic_to_waveform_new(semantics, _speaker, waveform_temp, output_full=True,
                                                             progress=progress)
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.name = temp.name.replace(temp.name.replace('\\', '/').split('/')[-1], 'speaker.npz')
        numpy.savez(temp.name, **history_prompt)
        return (SAMPLE_RATE, audio), temp.name

    def unload_model(self):
        # from bark.generation import clean_models
        # clean_models()

        import bark.generation as bark_gen
        model_keys = list(bark_gen.models.keys())
        for k in model_keys:
            if k in bark_gen.models:
                del bark_gen.models[k]
        bark_gen._clear_cuda_cache()
        gc.collect()

    def load_model(self, progress=gradio.Progress()):
        from webui.modules.implementations.patches.bark_generation import preload_models_new
        gpu = not settings.get('bark_use_cpu')
        preload_models_new(
            text_use_gpu=gpu,
            fine_use_gpu=gpu,
            coarse_use_gpu=gpu,
            codec_use_gpu=gpu,
            progress=progress
        )


# class CoquiTTS(mod.TTSModelLoader):
#     no_install = True
#     model = 'Coqui TTS'
#
#     current_model: TTS = None
#     current_model_name: str = None
#
#     def load_model(self, progress=gradio.Progress()):
#         pass
#
#     def unload_model(self):
#         self.current_model_name = None
#         self.current_model = None
#         gc.collect()
#         torch.cuda.empty_cache()
#
#     def tts_speakers(self):
#         if self.current_model is None:
#             return gradio.update(choices=[]), gradio.update(choices=[])
#         speakers = list(
#             dict.fromkeys([speaker.strip() for speaker in self.current_model.speakers])) if self.current_model.is_multi_speaker else []
#         languages = list(dict.fromkeys(self.current_model.languages)) if self.current_model.is_multi_lingual else []
#         return gradio.update(choices=speakers), gradio.update(choices=languages)
#
#     def _components(self, **quick_kwargs):
#         with gradio.Row(visible=False) as r1:
#             selected_tts = gradio.Dropdown(ModelManager(models_file=TTS.get_models_file_path(), progress_bar=False, verbose=False).list_tts_models(), label='TTS model', info='The TTS model to use for text-to-speech',
#                                            allow_custom_value=True, **quick_kwargs)
#             selected_tts_unload = gradio.Button('💣', variant='primary tool offset--10', **quick_kwargs)
#
#         with gradio.Row(visible=False) as r2:
#             speaker_tts = gradio.Dropdown(self.tts_speakers()[0]['choices'], label='TTS speaker',
#                                           info='The speaker to use for the TTS model, only for multi speaker models.', **quick_kwargs)
#             speaker_tts_refresh = gradio.Button('🔃', variant='primary tool offset--10', **quick_kwargs)
#
#         with gradio.Row(visible=False) as r3:
#             lang_tts = gradio.Dropdown(self.tts_speakers()[1]['choices'], label='TTS language',
#                                        info='The language to use for the TTS model, only for multilingual models.', **quick_kwargs)
#             lang_tts_refresh = gradio.Button('🔃', variant='primary tool offset--10', **quick_kwargs)
#
#         speaker_tts_refresh.click(fn=self.tts_speakers, outputs=[speaker_tts, lang_tts])
#         lang_tts_refresh.click(fn=self.tts_speakers, outputs=[speaker_tts, lang_tts])
#
#         def load_tts(model):
#             if self.current_model_name != model:
#                 unload_tts()
#                 self.current_model_name = model
#                 self.current_model = TTS(model, gpu=True if torch.cuda.is_available() and settings.get('tts_use_gpu') else False)
#             return gradio.update(value=model), *self.tts_speakers()
#
#         def unload_tts():
#             if self.current_model is not None:
#                 self.current_model = None
#                 self.current_model_name = None
#                 gc.collect()
#                 torch.cuda.empty_cache()
#             return gradio.update(value=''), *self.tts_speakers()
#
#         selected_tts_unload.click(fn=unload_tts, outputs=[selected_tts, speaker_tts, lang_tts])
#         selected_tts.select(fn=load_tts, inputs=selected_tts, outputs=[selected_tts, speaker_tts, lang_tts])
#
#         text_input = gradio.TextArea(label='Text to speech text',
#                                      info='Text to speech text if no audio file is used as input.', **quick_kwargs)
#
#         return selected_tts, selected_tts_unload, speaker_tts, speaker_tts_refresh, lang_tts, lang_tts_refresh, text_input, r1, r2, r3
#
#
#
#     def get_response(self, *inputs, progress=gradio.Progress()):
#         selected_tts, selected_tts_unload, speaker_tts, speaker_tts_refresh, lang_tts, lang_tts_refresh, text_input = inputs
#         if self.current_model_name != selected_tts:
#             if self.current_model is not None:
#                 self.current_model = None
#                 self.current_model_name = None
#                 gc.collect()
#                 torch.cuda.empty_cache()
#             self.current_model_name = selected_tts
#             self.current_model = TTS(selected_tts, gpu=True if torch.cuda.is_available() and settings.get('tts_use_gpu') else False)
#         audio = np.array(self.current_model.tts(text_input, speaker_tts if self.current_model.is_multi_speaker else None, lang_tts if self.current_model.is_multi_lingual else None))
#         audio_tuple = (self.current_model.synthesizer.output_sample_rate, audio)
#         return audio_tuple, None


elements = []


def init_elements():
    global elements

    import webui.extensionlib.callbacks as cb
    extension_elements = []
    for el in cb.get_manager('webui.tts.list')():
        if isinstance(el, mod.TTSModelLoader):
            extension_elements.append(el)
        elif isinstance(el, list):
            extension_elements += el
    extension_elements = [e for e in extension_elements if isinstance(e, mod.TTSModelLoader)]  # Cleanup
    elements = [BarkTTS()] + extension_elements


def all_tts() -> list[mod.TTSModelLoader]:
    if not elements:
        init_elements()
    return elements


def all_elements(in_dict):
    l = []
    for value in in_dict.values():
        l += value
    return l


def all_elements_dict():
    d = {}
    for tts in all_tts():
        d[tts.model] = tts.gradio_components()
    return d
