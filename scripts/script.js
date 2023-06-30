// The main javascript
import {enableHorScrolling} from "./scrolltabs.js"

console.log(gradio_config)
window.addEventListener('load', () => {
    setTimeout(() => {
        enableHorScrolling()
    }, 2000) // Pray
})
