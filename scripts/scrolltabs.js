function scrollHorizontally(event, el) {
    event.preventDefault();
    const delta = Math.max(-1, Math.min(1, event.wheelDelta || -event.detail));
    el.scrollLeft -= delta * 40;
}

export function enableHorScrolling() {
    console.log(document.querySelectorAll('.tab-nav'))
    for (const tabselement of document.querySelectorAll('.tab-nav')) {
        tabselement.addEventListener('wheel', (e) => scrollHorizontally(e, tabselement))
    }
}