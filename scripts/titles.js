const titlesMap = {
    bycontent: {
        '🔃': 'Refresh',
        '🚀': 'Load',
        '💣': 'Unload'
    }
}

export function createTitles() {
    for (const content in titlesMap.bycontent) {
        for (const btn of document.getElementsByTagName('button')) {
            if (btn.innerText === content) {
                btn.title = titlesMap.bycontent[content]
            }
        }
    }
}

export function titles() {
    createTitles()
    document.addEventListener('change', () => {
        createTitles()
    })
}