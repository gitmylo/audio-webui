import argparse
import os

parser = argparse.ArgumentParser()

# Install
parser.add_argument('-si', '--skip-install', action='store_true', help='Skip installing packages')
parser.add_argument('-sv', '--skip-venv', action='store_true', help='Skip creating/activating venv, also skips install (for advanced users)')

# Models
# Bark
parser.add_argument('--bark-low-vram', action='store_true', help='Use low vram mode on bark')
parser.add_argument('--bark-cpu-offload', action='store_true', help='Use cpu offloading for lower vram usage on bark')
parser.add_argument('--bark-use-cpu', action='store_true', help='Use cpu on bark')
parser.add_argument('--bark-cloning-large-model', action='store_true', help='Use the larger voice cloning model for bark')

# TTS
parser.add_argument('--tts-use-cpu', action='store_true', help='Use cpu for tts instead of gpu')

# Gradio
parser.add_argument('-s', '--share', action='store_true', help='Share this gradio instance.')
parser.add_argument('-u', '--username', '--user', type=str, help='Gradio username')
parser.add_argument('-p', '--password', '--pass', type=str, help='Gradio password (defaults to "password")', default='password')
parser.add_argument('--theme', type=str, help='Gradio theme', default='gradio/soft')

args = parser.parse_args()

if args.bark_cpu_offload:
    os.environ['SUNO_OFFLOAD_CPU'] = "True"
