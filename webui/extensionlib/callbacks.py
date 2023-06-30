class CallBack:
    def __init__(self, priority, value):
        self.priority = priority
        self.callback = value

    def call(self, *args, **kwargs):
        self.callback(*args, **kwargs)


class CallBackManager:
    def __init__(self, name):
        self.name = name
        self.callbacks: list[CallBack] = []


callbacks: list[CallBackManager] = []


def by_name(name):
    matches = [callback for callback in callbacks if callback.name.casefold() == name.casefold()]
    if len(matches) == 0:
        return None
    return matches[0]
