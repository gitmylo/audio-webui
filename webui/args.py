import argparse

parser = argparse.ArgumentParser()

# Install
parser.add_argument('-si', '--skip-install', action='store_true', help='Skip installing packages')

# Gradio
parser.add_argument('-s', '--share', action='store_true', help='Share this gradio instance.')
parser.add_argument('-u', '--username', '--user', type=str, help='Gradio username')
parser.add_argument('-p', '--password', '--pass', type=str, help='Gradio password (defaults to "password")', default='password')

args = parser.parse_args()
