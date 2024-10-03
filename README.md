# Organ-Segmentation
## Deep learning multi modality automated organ segmentation tools.
This repsotiry is about multiple organ segmentation on multi modality images. 
all of the models were trained using nnU-Net right now so you can use inference instruction available at https://github.com/MIC-DKFZ/nnUNet after downloading the trained models folder. You can find inference instruction in example.py file provided as well. It contains easy to use inference codes with breaking down large images if needed and more useful options facilitationg inference on larger datasets. right now the code is supprting only NIFTI file format, you need to convert your dicom images to NIFTI before running the codes. The instruction and codes to convert dicom to nifti files would be available shortly. You can download the trained models here: https://drive.google.com/drive/folders/1R6_EELnOeTb27YfueGm-X-bR-glgqOAF?usp=drive_link

Besides you can download separated models for multiple modalities here:
Normal CT scan mullti organ segmentation: https://drive.google.com/drive/folders/1ltmjbqfoCBzPCIeh6FgxGlpcAFYHn6eA?usp=drive_link

low dose and ultra low dose CT scan multiple organ segmentation: https://drive.google.com/drive/folders/1Iux0_V4T9xMoeq5kLz9PuuBJxd4gJO51?usp=drive_link

18FDG PET multiple organ segmentation: https://drive.google.com/drive/folders/1UDmEn4ypkXhB9B38QrREcRn8Kmr_n9t2?usp=drive_link

68-Ga-PSMA PET multiple organ segmentation: https://drive.google.com/drive/folders/1bFhIrHMcLpobTvqHj1thQ6JN_ARTfSIi?usp=drive_link

Multi tracer PET cardiac and cradiac substructure segmentation: https://drive.google.com/drive/folders/1Y0Yh2YyjuWudaCIJwZkMvqWvVZlUew14?usp=drive_link


If you use these models in your research or project, please cite the relevant papers:

For CT Modalities: https://www.medrxiv.org/content/10.1101/2023.10.20.23297331v1

For PET Modalities: https://www.medrxiv.org/content/10.1101/2024.08.27.24312482v1

## Installation:
Make sure you have installed your GPU drivers and installed nnunetv2 as described in their instruction available at: https://github.com/MIC-DKFZ/nnUNet or using these commands:
pip install --upgrade https://github.com/MIC-DKFZ/nnUNet.git
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
