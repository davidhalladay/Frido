from setuptools import setup, find_packages

setup(
    name='frido',
    version='1.0.0',
    author='wancyuan',
    author_email='wancyuan@ntu.edu.tw',
    description='Feature Pyramid Diffusion for Complex Scene Image Synthesis',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
)