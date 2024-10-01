from setuptools import setup, find_packages

setup(
    name='inference_code',
    version='1.0',
    packages=find_packages(),
    install_requires=[glob, SimpleITK, termcolor, tqdm, shutil, pandas, multiprocessing, natsort, nnunetv2],  # Add any dependencies here
    entry_points={
        'console_scripts': [
            'run_inference=inference.file1:main',  # Maps command to script
        ],
    },
)
