from organsegment import predict_multi
from multiprocessing import freeze_support
import os

# List of NIfTI files (.nii.gz) for processing
list_images = []

if __name__ == "__main__":
    freeze_support()
    predict_multi(
        list_images=list_images,
        target="inplace",  # 'inplace' writes the segmentation output in the same folder as the image. You can also specify a custom path.
        root_folder=r"C:\yzdn\nnunet_completed",  # Path to the folder containing the nnUNet_trained_models directory.
        raw_temp_folder=os.environ.get('TEMP') or os.environ.get('TMP') or '/tmp',  # Temporary folder for intermediate data; SSD is recommended for speed.
        modality="ULDCT",  # Select modality from ["HQCT_SingleModel", "HQCT_MultiModel", "ULDCT", "GAPET", "FDGPET", "CardiacPET"].
        organ_names="all",  # Specify individual organs (e.g., "liver", "bones") or use "all" for full organ segmentation.
        config="3d_fullres",  # Configuration setting (currently, "3d_fullres" is the only available option).
        num_worker_preprocessing=8,  # Adjust this based on the RAM available on your system.
        num_worker_saving=8,  # Adjust this based on the RAM available on your system.
        overwrite=True,  # If True, previously generated segmentation files will be overwritten.
        use_folds=(0, 1, 2, 3, 4),  # Recommended to use all folds for better accuracy, but you can select specific folds (e.g., (0, 1)).
        device="cuda",  # CUDA (GPU) is recommended for inference; CPU usage is not advised.
        perform_everything_on_gpu=True,  # Performs all operations on GPU for faster processing.
        clean_before=True,  # Cleans up any existing files in the temp folder before running.
        clean_after=False,  # If True, removes temporary segmentation files after processing.
        test_validity_of_data=True,  # Ensures that each input image is valid and readable.
        tile_step_size=0.5,  # Sliding window inference overlap; larger values make inference faster (default is recommended).
        use_gaussian=True,  # Gaussian post-processing; refer to nnUNet documentation for more details.
        use_mirroring=False,  # Enables test-time augmentation, which may increase inference time.
        allow_tqdm=False,  # Disables the progress bar.
        verbose=True,  # Enables verbose output for detailed logging.
        move_incomplete_segments=True,  # Moves incomplete segmentations in case of an error (useful for large datasets).
        break_down_large_images=True,  # Enables breaking down large images to handle them efficiently on machines with limited RAM.
        crop_to_foreground=True,  # Crops the image to the body contour before further processing.
        treshold_cm=200,  # If images are larger than 200 cm in the cranio-caudal direction, they will be split for faster processing. 200 cm means no splitting.
        num_workers_breaking=8,  # Number of workers for breaking large images; adjust based on your machine’s capacity.
        overwrite_ensembling=False,  # Controls whether existing ensemble segmentations should be overwritten.
        remove_broken_images_at_end=True,  # Deletes broken/incomplete images after inference.
        ensemble_every_organ_as_completed=False,  # Ensemble organs as they complete rather than waiting for the entire dataset to finish.
        long_image_paths=False,  # Useful on Windows for handling long file paths.
    )
