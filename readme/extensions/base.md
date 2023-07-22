# Extension base

## What is this about?
The extension base shows the files that the extension loader will check for.

## Structure

Extensions are placed in the `extensions` folder.

2 different descriptions below, same structure, pick the one you understand best.
<details>
<summary>
As markdown list
</summary>

* extension
  * extension.json **(required)**
  * main.py
  * requirements.py
  * style.py
  * scripts
    * script.js
</details>

<details>
<summary>
As file path list
</summary>

extension/extension.json **(required)**  
extension/main.py  
extension/requirements.py  
extension/style.py  
extension/scripts/script.js
</details>

## File contents

extension/requirements.py: Refer to [Requirements docs](requirements.md)  
extension/style.py: Refer to [SimpleStyle docs](style.md)  
extension/scripts/script.js: Refer to [Custom javascript docs](js.md)

extension/extension.json:
```json
{
  "name": "Extension name",
  "description": "This extension does nothing.",
  "author": "GitMylo",
  "tags": []
}
```

extension/main.py:
```python
# Put any code here, such as callback registers etc. This code will be ran after install, on plugin init.
```