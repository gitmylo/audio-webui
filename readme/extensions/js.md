# Custom javascript

## What is it for?
If you want to add new features to the webui on the browser side. You can use javascript for that.

All javascript scripts are module scripts, so they can use `import`.

## How to use custom javascript
2 different descriptions below, same structure, pick the one you understand best.  
File placement:
<details>
<summary>
As markdown list
</summary>

* extension
  * extension.py
  * main.py
  * requirements.py
  * style.py
  * scripts
    * **script.js** <--
</details>

<details>
<summary>
As file path list
</summary>

extension/extension.py  
extension/main.py  
extension/requirements.py  
extension/style.py  
extension/scripts/**script.js** <--
</details>

```js
alert('Javascript from plugin!');
```

## Multiple scripts
extension/scripts/script.js
```js
// Regular import
import {alertFromImport} from './example.js'; // Make sure you include `.js`
alertFromImport();

// Import as different name
import {alertFromImport as importAlert} from './example.js';
importAlert();
```

extension/scripts/example.js
```js
// Using export declaration
export function alertFromImport() {
    alert('Javascript from import!');
}

// Using export list
function alertFromImport() {
    alert('Javascript from import!');
}

// Now export like
export {alertFromImport};
// Alternatively export with different name
export {alertFromImport as importAlert};
```
