from setuptools import setup, find_packages

setup(
    name="organsegment",  # Replace with your package name
    version="0.1.0",
    author="Yazdan Salimi",
    author_email="salimiyazdan@gmail.com",
    description="Multi Modality Organ Segmentation tools",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YazdanSalimi/Organ-Segmentation",
    packages=find_packages(),
    py_modules=["predict_multi", "ImageUtilities"],
    install_requires=[
        # List your dependencies here
        # e.g., 'numpy', 'pandas'
        "shutils", "pandas", "tqdm",
        "termcolor", "glob2", "torch",
        "package_name @ git+https://github.com/MIC-DKFZ/nnUNet.git",
        "package_name @ git+https://github.com/FabianIsensee/hiddenlayer.git",
        "SimpleITK", "multiprocess",
        "numpy", "natsort", "batchgenerators",        
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # You can change the license if necessary
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the Python version
)
