import gradio as gr
from .tabs import *

tabs_el: gr.Tabs = None


def create_ui(theme) -> gr.Blocks:

    css = """
    .gradio-container {
        max-width: calc(100% - 100px) !important;
    }
    
    .tabitem:not(.tabitem>.gap>.tabs>.tabitem) {
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
    
    .offset--10 {
        transform: translateY(-10px);
    }
    """

    tabs = [
        ('Text to speech', text_to_speech),
        ('RVC', rvc),
        ('AudioLDM', audioldm_tab),
        ('Whisper', whisper),
        ('Utils', utils_tab),
        ('Extra', extra_tab)
    ]
    global tabs_el
    with gr.Blocks(theme=theme, title='Audio WebUI', css=css) as webui:
        with gr.Tabs() as tabs_element:
            tabs_el = tabs_element
            for name, content in tabs:
                with gr.Tab(name, id=name):
                    content()
    return webui
