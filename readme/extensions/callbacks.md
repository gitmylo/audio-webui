# Extension callbacks
If you find something that you'd want a callback in, please suggest it on discord or create an issue.

## Hooking callbacks
Hooking callbacks is the main use for callbacks from the extension developer's side, callbacks are basically events,
which can easily be hooked.

You can hook a callback like this:
```python
# Define a function for the callback to run
def callback_function(*args, **kwargs):
    print('Callback was triggered!')

# Import the callback script
import webui.extensionlib.callbacks as cb

# Register callback through manager
callback = cb.CallBack(callback_function)
cb.get_manager('webui.init').register(callback)

# Register by name in a single line
callback = cb.register_by_name('webui.init', callback_function)
```

And you can remove a callback if needed, assuming you have a reference to the registered callback, like this:
```python
import webui.extensionlib.callbacks as cb

# Remove callback through manager
cb.get_manager('webui.init').unregister(callback)

# Remove callback through function
cb.unregister_by_name('webui.init', callback)
```

## Registering new callbacks (callback managers)
You can register callbacks through extensionlib's callbacks manager. Use this if you want users to be able to expand
your extension.  
(For example, an API extension which has a callback to register additional routes.)  
(Not required, as callbacks will be auto-registered if missing)

```python
import webui.extensionlib.callbacks as cb

new_manager = cb.register_new('example_extension.example')

manager_from_name = cb.get_manager('example_extension.example')

assert new_manager == manager_from_name
# These 2 will be the same,
# unless there was already a callback registered with the same name

# You can activate a callback like this:
new_manager()

# You can supply parameters too
new_manager('Positional', 2, example='keyword')

# You can access outputs as an array
outputs = new_manager()
# [] - no callbacks were ran as none were registered
# This will store raw return values from the callbacks, make sure to check them.
```

## List of callbacks
Indents get replaced with ".", so for example:
* example
  * indented (example.indented)
  * indented2 (example.indented2)
  * *non-callback*
    * callback (example.non-callback.callback)

* webui
  * init [Called when the webui is initiated, after extensions have been initialized.]
  * [settings](callbacks/webui/settings.md) [Called to fetch settings for this extension.]
  * [tabs](your_first_extension.md) [Called after the base tabs have been registered, allows you to add new tabs]
    * utils [Same as tabs, but on the utils tab]
  * *tts*
    * [list](callbacks/webui/tts/list.md) [Called on tts list init, register new tts modules here]

Please request new callbacks if you need them, don't monkeypatch them in, that might break compatibility with updates and other extensions.
