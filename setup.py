import os
from setuptools import setup, find_packages

def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements

requirements = resolve_requirements(
    os.path.join(os.path.dirname(__file__), 'requirements.txt')
)

setup(
    name='transoar',
    version='0.1',
    description='Medical Vision Transformer Library.',
    author='Bastian Wittmann',
    author_email='bastian.wittmann@tum.de',
    url='https://github.com/bwittmann/transoar',
    packages=find_packages(),
    entry_points={},
    install_requires=requirements,
    python_requires=">=3.8"
)