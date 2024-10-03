from organsegment import predict_multi
from multiprocessing import freeze_support
import os
list_images = [] # list of nifti files (.nii.gz)
if __name__ == "__main__":
    freeze_support()
    predict_multi(list_images,
                    target = "inplace", # inplace means a file with segmentation names will be written in the same folder as image, this could be a path to a folder you want to have the segmentation files
                    root_folder = r"C:\yzdn\nnunet_completed", # this is a folder you have pasted nnUNet_trained_models folder
                    raw_temp_folder = os.environ.get('TEMP') or os.environ.get('TMP') or '/tmp', #temp path to save intermediate and processed data suggested to be on SSD drive
                    modality = "ULDCT", # ["HQCT_SinlgeModel", "HQCT_MultiModel", "ULDCT", "GAPET", "FDGPET", "CardiacPET"]
                    organ_names = "all", # you can select single organs such as bones or liver
                    config = "3d_fullres", # this is the only configuration avaiable now
                    num_worker_preprocessing = 8, # choose according to amount of RAM you have on your machine
                    num_worker_saving = 8, # choose according to amount of RAM you have on your machine
                    overwrite = True, # overwrite the previously generated segmentation in target fodler
                    use_folds =  (0,1,2,3,4), # suggested to ensemble all folds for better results but you can selcte as (0,1,)
                    device = "cuda", # doing inference on CPU is not recommended
                    perform_everything_on_gpu = True,
                    clean_before = True, 
                    clean_after = False, # this option removes the segmentations generated in temp folder
                    test_validity_of_data = True, # test every image to ensure the image is readable and correct
                    tile_step_size = .5, # sliding window inference overlap, larger means faster, default value is recommended
                    use_gaussian = True, # refer to nnunet for details
                    use_mirroring = False, # test time augmentation which increases the inference time
                    allow_tqdm = False, 
                    verbose = True,
                    move_incomplete_segments = True, # this options is more usefull when you do inference on a large dataset, in case of error during inference it transfer the partial incomplete data
                    break_down_large_images = True, # this option is useful if you have limited ram on your machine and dealing with total body big images, it has two steps of croppping images to body contour and breaking down big images
                    crop_to_foreground = True,
                    treshold_cm = 200, # images larger than this size (cm) in cranio caudal direction will be broken to multiple parts to increase speed. default value of 200 cm means not breaking down images.
                    num_workers_breaking = 8, # number of multip processing workers to crop images to body contour. choose it according to your machine
                    overwrite_ensembling = False,
                    remove_broken_images_at_end = True,
                    ensemble_every_organ_as_completed = False,
                    long_image_paths = False, # this is useful if you use windows and the image address is too long
                    )    
