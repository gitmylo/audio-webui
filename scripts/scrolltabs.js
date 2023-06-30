function scrollHorizontally(event, el) {
    event.preventDefault();
    const delta = Math.max(-1, Math.min(1, event.wheelDelta || -event.detail));
    el.scrollLeft -= delta * 40;
    scrollGlow(el)
}

function scrollGlow(el) {
    const leftMax = el.scrollLeftMax
    const leftVal = el.scrollLeft

    let left = false
    let right = false

    if (leftMax !== 0) {
        if (leftVal > 0) {
            left = true
        }
        if (leftVal < leftMax-5) {
            right = true
        }
    }

    el.classList.toggle('leftscroll', left)
    el.classList.toggle('rightscroll', right)
}

export function enableHorScrolling() {
    for (const tabselement of document.querySelectorAll('.tab-nav')) {
        tabselement.addEventListener('wheel', (e) => scrollHorizontally(e, tabselement))
        scrollGlow(tabselement)
    }
    window.addEventListener('resize', () => {
        for (const tabselement of document.querySelectorAll('.tab-nav')) {
            scrollGlow(tabselement)
        }
    })
}