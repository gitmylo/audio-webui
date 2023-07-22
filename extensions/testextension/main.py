import webui.extensionlib.callbacks as cb


def init_callback():
    print('All extensions have initialized!')


cb.register_by_name('webui.init', init_callback)

cb.register_by_name('webui.settings', lambda: {
    'test_setting': {
        'tab': '🤖 Test extension',
        'type': bool,
        'default': False,
        'readname': 'Test setting from test extension',
        'description': 'This setting is loaded from the webui.settings event, It\'s actually part of a lambda expression'
    }
})
