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
(Example: An API extension which has a callback to register additional routes.)

main.py
```python
import webui.extensionlib.callbacks as cb

new_manager = cb.register_new('example_extension.example')

manager_from_name = cb.get_manager('example_extension.example')

assert new_manager == manager_from_name
# These 2 will be the same,
# unless there was already a callback registered with the same name
```

## List of callbacks
TODO
