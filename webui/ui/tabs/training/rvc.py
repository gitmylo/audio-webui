import gradio
import webui.ui.tabs.training.training.rvc_workspace as rvc_ws


def change_setting(name, value):
    rvc_ws.current_workspace.data[name] = value
    rvc_ws.current_workspace.save()


def train_rvc():
    with gradio.Row():
        with gradio.Column():
            gradio.Markdown('''
            # Workspaces
            ''')
            with gradio.Tabs():
                with gradio.Tab('Load'):
                    with gradio.Row():
                        workspace_select = gradio.Dropdown(rvc_ws.get_workspaces(), label='Select workspace')
                        refresh_workspaces = gradio.Button('🔃', variant='primary tool offset--10')
                with gradio.Tab('Create'):
                    create_name = gradio.Textbox(label='Name')
                    version_sample_rate = gradio.Radio(['v1 40k', 'v1 48k', 'v2 40k'], value='v2 40k', label='version and sample rate')
                    create_button = gradio.Button('Create workspace', variant='primary')
        with gradio.Column(visible=False) as settings:
            gradio.Markdown('''
            # Settings
            ''')
            with gradio.Tabs():
                with gradio.Tab('data'):
                    dataset_path = gradio.Textbox(label='Dataset path', info='The path to the dataset containing your training audio.')
                    dataset_path.change(fn=lambda val: change_setting('dataset', val), inputs=dataset_path)
                    process_dataset = gradio.Button('Resample and split dataset', variant='primary')
                    f0_method = gradio.Radio(["none", "dio", "pm", "harvest", "torchcrepe", "torchcrepe tiny"], value='harvest', label='Pitch extraction method', info='Harvest is usually good, crepe has potential to be even better.')
                    f0_method.change(fn=lambda val: change_setting('f0', val), inputs=f0_method)
                    pitch_extract = gradio.Button('Extract pitches', variant='primary')
                with gradio.Tab('train'):
                    pass
                with gradio.Tab('how to?'):
                    # TODO: remove Not implemented yet once implemented
                    gradio.Markdown('''
                    # Not implemented yet...
                    ## How to train
                    1. Collect audio data (if from youtube, you can use the Utils tab to download audio from youtube quickly).
                      * Optional: use Utils tab to split vocals from music if there's background music.
                    2. Open the "data" tab.
                        1. Put the path to the folder containing your audio data as the training path.
                        2. Click "Process dataset"
                        3. Pick your preferred pitch extraction method. Harvest and crepe are recommended.
                        4. Click "Extract pitches"
                    3. Open the "train" tab.
                        1. Set training epochs.
                            * You can click the "use suggested" button to set your training epochs to the recommended amount based on how much data you have.
                        2. Click "Train", and wait for training to complete.
                    4. You now have a trained model in your 'data/training/RVC/{workspace}/models' folder.
                    ''')
        with gradio.Column():
            gradio.Markdown('''
            # Status
            ''')
            status_box = gradio.TextArea(label='Status')

    def load_workspace(name):
        rvc_ws.current_workspace = rvc_ws.RvcWorkspace(name).load()
        ws = rvc_ws.current_workspace
        return f'Loaded workspace {name}', ws.name, gradio.update(visible=True), ws.data['dataset'], ws.data['f0']

    def list_workspaces():
        return gradio.update(choices=rvc_ws.get_workspaces())

    def create_workspace(name, vsr):
        rvc_ws.current_workspace = rvc_ws.RvcWorkspace(name).create({
            'vsr': vsr
        })
        rvc_ws.current_workspace.save()
        return load_workspace(name)

    setting_elements = [status_box, workspace_select, settings, dataset_path, f0_method]

    process_dataset.click(fn=rvc_ws.process_dataset, outputs=status_box)
    pitch_extract.click(fn=rvc_ws.pitch_extract, outputs=status_box)

    workspace_select.select(fn=load_workspace, inputs=workspace_select, outputs=setting_elements, show_progress=True)
    refresh_workspaces.click(fn=list_workspaces, outputs=workspace_select, show_progress=True)
    create_button.click(fn=create_workspace, inputs=[create_name, version_sample_rate], outputs=setting_elements, show_progress=True)
