import argparse
import os


class BarkModelChoices(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if len(values) != 3:
            raise ValueError('Incorrect syntax for --bark-models-mix, use 3 characters')
        valid_models = {
            'l': {
                'name': 'Large',
                'large': True
            },
            's': {
                'name': 'Small',
                'large': False
            }
        }
        selected_models = []
        for char in values:
            if char not in valid_models.keys():
                raise ValueError(f'An unknown model was specified for --bark-models-mix, Available models: {list(valid_models.keys())}')
            selected_models.append(valid_models[char])
        model_indexes = ['text', 'coarse', 'fine']
        setattr(args, self.dest, {key: value for key, value in zip(model_indexes, selected_models)})


parser = argparse.ArgumentParser(prog='Audio-Webui', description='A webui for audio related neural networks.')

# Install
parser.add_argument('-si', '--skip-install', action='store_true', help='Skip installing packages')
parser.add_argument('-sv', '--skip-venv', action='store_true', help='Skip creating/activating venv, also skips install (for advanced users)')
parser.add_argument('--no-data-cache', action='store_true', help='Don\'t override the default huggingface_hub cache path.')

# Models
# Bark
bark_models = parser.add_mutually_exclusive_group()
bark_models.add_argument('--bark-use-small', action='store_true', help='Use low vram mode on bark')
bark_models.add_argument('--bark-models-mix', action=BarkModelChoices, help='Mix bark small (s) and large (l) models, example: "lsl" for large text, small coarse, large fine')

parser.add_argument('--bark-half', action='store_true', help='Lower vram usage through half precision.')
parser.add_argument('--bark-cpu-offload', action='store_true', help='Use cpu offloading for lower vram usage on bark')
parser.add_argument('--bark-use-cpu', action='store_true', help='Use cpu on bark')

# TTS
parser.add_argument('--tts-use-gpu', action='store_true', help='Use gpu for TTS')

# Gradio
parser.add_argument('-s', '--share', action='store_true', help='Share this gradio instance.')
parser.add_argument('-u', '--username', '--user', type=str, help='Gradio username')
parser.add_argument('-p', '--password', '--pass', type=str, help='Gradio password (defaults to "password")', default='password')
parser.add_argument('--theme', type=str, help='Gradio theme', default='gradio/soft')
parser.add_argument('-l', '--listen', action='store_true', help='Listen on 0.0.0.0')
parser.add_argument('--port', type=int, help='Use a different port, automatic when not set.', default=None)

# Visualizer
parser.add_argument('--wav-type', type=str, choices=['none', 'gradio', 'showwaves'], default='gradio', help='The type of waveform visualizer to use')

args = parser.parse_args()

if args.bark_cpu_offload:
    os.environ['SUNO_OFFLOAD_CPU'] = "True"
