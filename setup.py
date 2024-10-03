from setuptools import setup, find_packages

setup(
    name="Organ-Segmentation",  # Replace with your package name
    version="0.1.0",
    author="Yazdan Salimi",
    author_email="salimiyazdan@gmail.com",
    description="Multi Modality Organ Segmentation tools",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YazdanSalimi/Organ-Segmentation",
    packages=find_packages(),
    py_modules=["inference", "ImageUtilities"],
    install_requires=[
        # List your dependencies here
        # e.g., 'numpy', 'pandas'
        "shutils", "pandas", "tqdm",
        "termcolor", "glob", "torch",
        "nnunetv2", "SimpleITK", "multiprocessing",
        "numpy", "natsort",        
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # You can change the license if necessary
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the Python version
)
