# SimpleStyle

## What is it?
SimpleStyle is a simple way to write css in python, it allows you to give your styles a priority compared to other
SimpleStyle scripts. It uses `__enter__` and `__exit__` to allow for easy styling, in a readable way for python.

If you have any suggestions or issues, please create an issue or send a message on discord.

## Example
Example CSS:
```css
.classname:not(.classname2) {
    border: 1px solid black;
    position: relative;
}
#id {
    color: white;
}
```

SimpleStyle equivalent:
```python
from simplestyle.manager import SimpleStyle, StyleRule, StyleValue
with SimpleStyle(priority=0):  # 0 is default, higher is later
    with StyleRule('.classname:not(.classname2)'):  # Selector
        StyleValue('border', '1px solid black')  # Value
        StyleValue('position', 'relative')
    with StyleRule('#id'):
        StyleValue('color', 'white')
```
(Make sure to put your style code in extension/**style.py** for it to be recognised)
