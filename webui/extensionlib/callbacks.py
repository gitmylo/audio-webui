class CallBack:
    def __init__(self, value, priority=0):
        self.priority = priority
        self.callback = value

    def call(self, *args, **kwargs):
        return self.callback(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class CallBackManager:
    def __init__(self, name):
        self.name = name
        self.callbacks: list[CallBack] = []

    def register(self, callback: CallBack):
        self.callbacks.append(callback)
        self.callbacks.sort(key=lambda c: c.priority, reverse=True)

    def unregister(self, callback: CallBack):
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            return True
        return False

    def call(self, *args, **kwargs):
        return [cb(*args, **kwargs) for cb in self.callbacks]

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


callbacks: list[CallBackManager] = []


def get_manager(name) -> CallBackManager | None:
    """Get a callback manager by its registered name. (case insensitive)"""
    matches = [callback for callback in callbacks if callback.name.casefold() == name.casefold()]
    if len(matches) == 0:
        return register_new(name)
    return matches[0]


def register_by_name(name: str, callback, priority: int = 0) -> CallBack | None:
    """Get and register a callback."""
    callback = CallBack(callback, priority)
    manager = get_manager(name)
    if not manager:
        return None
    manager.register(callback)
    return callback


def unregister_by_name(name: str, callback: CallBack) -> bool:
    """Unregister a callback based on its name."""
    manager = get_manager(name)
    if not manager:
        return False
    return manager.unregister(callback)


def register_new(name: str) -> CallBackManager:
    """
    Please don't use duplicates.

    Please use "." to split your callback names instead of spaces.
    """
    callback = CallBackManager(name)
    callbacks.append(callback)
    return callback


def get_callbacks() -> list[str]:
    """Returns a list of all callback names."""
    return [callback.name for callback in callbacks]

