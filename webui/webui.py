from webui.ui.ui import create_ui
from .args import args
import gradio


def launch_webui():
    auth = (args.username, args.password) if args.username else None

    template_response_original = gradio.routes.templates.TemplateResponse

    # Magic monkeypatch
    def template_response(*args, **kwargs):
        res = template_response_original(*args, **kwargs)
        res.body = res.body.replace(b'</body>',
                                    f'<script type="module" src="file=scripts/script.js"></script></body>'.encode("utf8"))
        res.init_headers()
        return res
    gradio.routes.templates.TemplateResponse = template_response



    create_ui(args.theme).queue().launch(share=args.share,
                                         auth=auth,
                                         server_name='0.0.0.0' if args.listen else None,
                                         server_port=args.port,
                                         favicon_path='assets/logo.png',
                                         inbrowser=args.launch)
