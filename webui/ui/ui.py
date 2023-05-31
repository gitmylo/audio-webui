import gradio as gr
from .tabs import *


def create_ui(theme) -> gr.Blocks:

    css = """
    .gradio-container {
        max-width: calc(100% - 100px) !important;
    }
    
    .tabitem {
        height: calc(100vh - 100px) !important;
        overflow: auto;
    }
    
    .tool{
        max-width: 2.2em;
        min-width: 2.2em !important;
        height: 2.4em;
        align-self: end;
        line-height: 1em;
        border-radius: 0.5em;
    }
    
    .smallsplit {
        max-width: 2.2em;
        min-width: 2.2em !important;
        align-self: end;
        border-radius: 0.5em;
    }
    """

    tabs = [
        ('Text to speech', text_to_speech),
        ('rvc', rvc),
        ('extra', extra_tab)
    ]
    with gr.Blocks(theme=theme, title='Audio WebUI', css=css) as webui:
        with gr.Tabs():
            for name, content in tabs:
                with gr.Tab(name):
                    content()
    return webui
