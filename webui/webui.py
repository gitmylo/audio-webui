from webui.ui.ui import create_ui
from .args import args


def launch_webui():
    auth = (args.username, args.password) if args.username else None
    create_ui(args.theme).queue().launch(share=args.share,
                                         auth=auth,
                                         server_name='0.0.0.0' if args.listen else None,
                                         server_port=args.port)
