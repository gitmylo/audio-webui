// The main javascript
import {enableHorScrolling} from "./scrolltabs.js"
import {titles} from "./titles.js"

window.addEventListener('load', () => {
    setTimeout(() => {
        enableHorScrolling()
        titles()
    }, 2000) // Pray
})
