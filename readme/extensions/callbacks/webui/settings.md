# webui.settings callback

### Description:
The settings callback allows you to register settings in the settings tab on audio-webui


### Usage
(lambda recommended, but not necessary)
```python
import webui.extensionlib.callbacks as cb

cb.register_by_name('webui.settings', lambda: {
    'ExampleSetting': {
        'tab': '🤯 Custom settings',  # Optional, if not set, "other" tab is used.
        'type': bool,
        'default': False,
        'readname': 'Example setting',
        'description': 'This is an example setting.'
    }
})
```

Values of settings can be checked by doing:
```python
import webui.ui.tabs.settings as settings

example_value = settings.get('ExampleSetting')
```

### Custom settings
To create custom settings, create a class which inherits `CustomSetting` from `webui.ui.tabs.settings`.

A customsetting has a `create_ui` function, `load_val` and `save_val`.

They will automatically be saved and loaded using the values from `load_val` and `save_val`.

`save_val`: returns the object to save, must be json serializable, examples: `str`, `int`, `float`, `dict`, `list`.

`load_val`: is given the value to load `val`, and returns the value to set for `self.value` when the `CustomSetting` is instantiated.

You can look inside of `webui/ui/tabs/settings.py` for a practical example usage of customsetting.
