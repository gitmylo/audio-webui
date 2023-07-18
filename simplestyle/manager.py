class RawStyleValue:
    def __init__(self, value):
        stack[-1].stack[-1].values.append(value)


class StyleValue(RawStyleValue):
    def __init__(self, key, value):
        super().__init__(f'{key}: {value};')


class StyleRule:
    def __init__(self, selector):
        self.selector = selector
        self.values = []

    def __enter__(self):
        stack[-1].stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        stack[-1].rules.append(stack[-1].stack.pop())


class SimpleStyle:
    def __init__(self, priority=0):
        self.priority = priority
        self.stack = []
        self.rules = []

    def __enter__(self):
        stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        rules.append(stack.pop())


stack: list[SimpleStyle] = []
rules: list[SimpleStyle] = []


def create_stylesheet():

    def sort_key(value):
        return value.priority

    full_sheet = ''
    rules.sort(key=sort_key)
    for rule in rules:
        for stylerule in rule.rules:
            full_sheet += f'{stylerule.selector} {{'
            for stylevalue in stylerule.values:
                full_sheet += stylevalue
            full_sheet += '}'
    return full_sheet
