# Organ-Segmentation
## Deep Learning Multi-Modality Automated Organ Segmentation Tools
This repsotiry is about multiple organ segmentation on multi modality images. 
This repository provides tools for automated segmentation of multiple organs across various imaging modalities using deep learning. All models have been trained using nnU-Net. You can find inference instructions in the provided inference-example.py file. It includes easy-to-use code for handling large images and several useful options for efficient inference on large datasets. You can also select the appropriate input image modality based on your needs. 
Currently, the code supports only the NIfTI file format, so you’ll need to convert your DICOM images to NIfTI before running the code. Instructions and scripts for converting DICOM to NIfTI will be available shortly. Alternatively, you can follow the standard nnU-Net inference instructions available on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) after downloading the trained models folder.
## Available Models

We offer two categories of CT segmentation models:

HQCT_SingleModel: Suitable for all CT images with acceptable image quality.
ULDCT: Dedicated to ultra-low-dose CT images, such as those acquired in for ultra low dose PET/CT, SPECT/CT for any other ultra low dose CT.
For PET images, the models require body-weight SUV unit images as input. Separate models are available for Ga-PSMA and 18FDG PET images. Please ensure you select the appropriate model based on your input, which may be an attenuation and scatter-corrected PET image or a non-corrected PET image.


## Download Trained Models


[All trained models](https://drive.google.com/drive/folders/1R6_EELnOeTb27YfueGm-X-bR-glgqOAF?usp=drive_link)

## Separate Models by Modality:


[Normal CT Scan Multi-Organ Segmentation](https://drive.google.com/drive/folders/1ltmjbqfoCBzPCIeh6FgxGlpcAFYHn6eA?usp=drive_link)

[Low-Dose and Ultra-Low-Dose CT Scan Multi-Organ Segmentation](https://drive.google.com/drive/folders/1Iux0_V4T9xMoeq5kLz9PuuBJxd4gJO51?usp=sharing)

[18FDG PET Multi-Organ Segmentation](https://drive.google.com/drive/folders/1UDmEn4ypkXhB9B38QrREcRn8Kmr_n9t2?usp=drive_link)

[68-Ga-PSMA PET Multi-Organ Segmentation](https://drive.google.com/drive/folders/1bFhIrHMcLpobTvqHj1thQ6JN_ARTfSIi?usp=drive_link)

[Multi-Tracer PET Cardiac and Cardiac Substructure Segmentation](https://drive.google.com/drive/folders/1Y0Yh2YyjuWudaCIJwZkMvqWvVZlUew14?usp=drive_link)

## Citation
If you use these models in your research or project, please cite the relevant papers:

[CT Modalities paper](https://www.medrxiv.org/content/10.1101/2023.10.20.23297331v1)

[PET Modalities paper](https://www.medrxiv.org/content/10.1101/2024.08.27.24312482v1)

## Installation:
Before running the tools, make sure your GPU drivers are properly installed and that nnunetv2 is installed as per their instructions, which can be found at: https://github.com/MIC-DKFZ/nnUNet.
Alternatively, you can install nnunetv2 using the following commands:
```bash
pip install --upgrade git+https://github.com/MIC-DKFZ/nnUNet.git

pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```
To install this repository, simply run:
```bash
pip install git+https://github.com/YazdanSalimi/Organ-Segmentation.git
```
We welcome any feedback, suggestions, or contributions to improve this project!

for any furtehr question please email me at: [salimiyazdan@gmail.com](mailto:salimiyazdan@gmail.com)
