# Your first audio-webui extension

Create a repository, or a regular folder, repositories will get automatic updates though.

Let's call it "example"

Now, lets create an **extension.json** at **example/extension.json**
```json
{
  "name": "example",
  "description": "An example extension for this tutorial.",
  "author": "GitMylo",
  "tags": []
}
```

Great! Now your extension can be loaded by audio webui, when it's in the `extensions` folder.

For this example, we will create an extension which adds a new tab to audio-webui

create a file at **example/main.py**
```python
import webui.extensionlib.callbacks as cb
import gradio

def new_tab():
    with gradio.Tab('Extension tab!'):
        gradio.Markdown('# This tab is created from an extension!')

cb.register_by_name('webui.tabs', new_tab)
```
