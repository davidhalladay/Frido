from setuptools import setup, find_packages

setup(
    name='frido',
    version='1.1',
    author='wancyuan',
    author_email='wancyuanf@gmail.com',
    description='Feature Pyramid Diffusion for Complex Scene Image Synthesis',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
)