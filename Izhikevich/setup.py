from setuptools import setup
with open('requirements.txt', 'r') as f:
    for line in f:
        setup(require = line)
