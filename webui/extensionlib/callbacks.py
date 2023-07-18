class CallBack:
    def __init__(self, priority, value):
        self.priority = priority
        self.callback = value

    def call(self, *args, **kwargs):
        self.callback(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class CallBackManager:
    def __init__(self, name):
        self.name = name
        self.callbacks: list[CallBack] = []

    def register(self, callback: CallBack):
        self.callbacks.append(callback)
        self.callbacks.sort(key=lambda c: c.priority, reverse=True)

    def call(self, *args, **kwargs):
        for cb in self.callbacks:
            cb(*args, **kwargs)


callbacks: list[CallBackManager] = []


def by_name(name):
    matches = [callback for callback in callbacks if callback.name.casefold() == name.casefold()]
    if len(matches) == 0:
        return None
    return matches[0]
