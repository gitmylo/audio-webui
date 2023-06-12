import json
import os


class Workspace:
    name: str
    data: dict
    base_path: str = ''

    def __init__(self, name, data=None):
        self.name = name
        self.data = {} if data is None else data

    @property
    def space_path(self):
        return os.path.join('data', 'training', self.base_path, self.name)

    @property
    def json_path(self):
        return os.path.join(self.space_path, 'workspace.json')

    def list_workspaces(self):
        directory = self.space_path
        os.makedirs(directory, exist_ok=True)
        return os.listdir(directory)

    def load(self):
        self.data = json.load(open(self.json_path, 'r'))
        return self

    def save(self):
        os.makedirs(self.space_path, exist_ok=True)
        json.dump(self.data, open(self.json_path, 'w'))
        return self

    def create(self, data):
        self.data = data
        return self
