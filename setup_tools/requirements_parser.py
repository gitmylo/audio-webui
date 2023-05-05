from .os import is_windows


def parse_requirements(req_file='install_requirements.txt'):

    # Some stuff for the install_requirements
    windows = is_windows()

    requirements_parsed = []
    with open(req_file, 'r') as file:
        for line in file.readlines():
            line = line.strip()

            # Make sure it can be unpacked
            semis = line.count(';')
            add_semis = (2 - semis) * ';'
            line = line + add_semis

            packages, args, condition = line.split(';')[:3]
            # if eval(condition):
            if condition:
                if eval(condition):
                    for package in packages.split(' '):
                        package = package.strip()
                        if package:
                            requirements_parsed.append(f'{package} {args}')
    return requirements_parsed
