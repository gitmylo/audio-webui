# Extension requirements

## What is it for?
Extension requirements are requirements for your extensions, if there's a python library you need,
but isn't in default audio-webui, you can add it using the `requirements.py`.

## Example:
extension/requirements.py
```python
from setup_tools.magicinstaller.requirement import SimpleRequirement, SimpleRequirementInit, CompareAction

class GitRequirementExample(SimpleRequirement):
    package_name = 'name'
    
    def is_right_version(self):
        return self.get_package_version('name') == 'gitcommithash'
    
    def install(self) -> tuple[int, str, str]:
        return self.install_pip('git+https://github.com/user/repo.git@gitcommithash', 'name')

def requirements():
    return [
        SimpleRequirementInit('name'), # Regular package
        SimpleRequirementInit('name', CompareAction.EQ, '1.5.4'), # Version specific
        GitRequirementExample() # Custom package
    ]
```
