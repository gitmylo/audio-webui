import os.path
import gradio


def get_gradio_readme(name):
    return open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'readme', 'gradio', name), 'r', encoding='utf8').read()


def info_tab():
    gradio.Markdown(get_gradio_readme('info.md'))
