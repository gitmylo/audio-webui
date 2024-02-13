import argparse


parser = argparse.ArgumentParser(prog='Audio-Webui', description='A webui for audio related neural networks.')

# Install
parser.add_argument('-si', '--skip-install', action='store_true', help='Skip installing packages')
parser.add_argument('-sv', '--skip-venv', action='store_true', help='Skip creating/activating venv, also skips install (for advanced users)')
parser.add_argument('--no-data-cache', action='store_true', help='Don\'t override the default huggingface_hub cache path.')
parser.add_argument('-v', '--verbose', action='store_true', help='Show more info, like logs during installs')

# Gradio
parser.add_argument('-s', '--share', action='store_true', help='Share this gradio instance.')
parser.add_argument('-u', '--username', '--user', type=str, help='Gradio username')
parser.add_argument('-p', '--password', '--pass', type=str, help='Gradio password (defaults to "password")', default='password')
parser.add_argument('--theme', type=str, help='Gradio theme', default='gradio/soft')
parser.add_argument('-l', '--listen', action='store_true', help='Listen on 0.0.0.0')
parser.add_argument('--port', type=int, help='Use a different port, automatic when not set.', default=None)
parser.add_argument('--launch', action='store_true', help='Automatically open a browser window when the webui launches.')

args = parser.parse_args()
