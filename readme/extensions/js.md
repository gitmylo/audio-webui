# Custom javascript

## What is it for?
If you want to add new features to the webui on the browser side. You can use javascript for that.

All javascript scripts are module scripts, so they can use `import`.

## How to use custom javascript
TODO: Plugin structure indicating **script.js** here

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
