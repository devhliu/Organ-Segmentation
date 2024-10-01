import sys; sys.path.append(r"path to source codes")
from ImageUtitlities import *

def predict_multi(list_images : list,
                target = "inplace",
                root_folder:str = r"C:\yzdn\nnunet_completed", 
                raw_temp_folder:str = r"path to 'organ-segmentation-models' folder", 
                modality = "HQCT_SinlgeModel",
                organ_names = "all", 
                config = "3d_fullres", 
                save_probs = False,
                num_worker_preprocessing = 8,
                num_worker_saving = 8,
                overwrite = True,
                use_folds =  (0,1,2,3,4),
                checkpoint_name = "checkpoint_final.pth",
                device = "cuda", 
                perform_everything_on_gpu = True,
                clean_before = True,
                clean_after = False,
                test_validity_of_data = True,
                minimum_number_of_slices = 5,
                tile_step_size = .5,
                use_gaussian=True,
                use_mirroring=True,
                allow_tqdm = True,
                verbose = True,
                move_incomplete_segments = True,
                sign_images = False,
                break_down_large_images = True,
                crop_to_foreground = True,
                treshold_cm = 300,
                overlap = 0, 
                num_workers_breaking = 8,
                overwrite_ensembling = False,
                remove_broken_images_at_end = True,
                ensemble_every_organ_as_completed = False,
                long_image_paths = False,
                ):
    from glob import glob
    import os
    import SimpleITK as sitk
    from termcolor import cprint
    from tqdm import tqdm
    import time
    import timeit
    import numpy as np
    import gc
    import torch
           
    if not modality in ["HQCT_SinlgeModel", "HQCT_MultiModel", "ULDCT", "GAPET", "FDGPET", "CardiacPET"]:
        cprint(f"Error, Modality valid values are : {["HQCT_SinlgeModel", "HQCT_MultiModel", "ULDCT", "GAPET", "FDGPET", "CardiacPET"]}", "white", "on_red")
        return "none"
    start_time = time.time()
    list_images = [x for x in list_images if not "--yzdnn.nii.gz" in os.path.basename(x) and not "--unified-segments" in os.path.basename(x)]
    list_images = [x for x in list_images if not "-nnbreak.nii.gz" in os.path.basename(x)]
    list_images = [x for x in list_images if not "-nncrop.nii.gz" in os.path.basename(x)]
    if test_validity_of_data:
        list_ct_url_pure = []
        for image_url_validation in tqdm(list_images, desc = "validating dataset", leave = False):
            try:
                image_temp = sitk.ReadImage(image_url_validation)
                if image_temp.GetDimension() > 2 and min(image_temp.GetSize()) > minimum_number_of_slices and not is_segment(image_temp)[-1]:
                    list_ct_url_pure.append(image_url_validation)
            except:
                print(f"this image contains bad data:   {image_url_validation}")
        cprint("Validation of data completed", "white", "on_cyan")        
        list_images = list_ct_url_pure 

    if break_down_large_images:
        cprint("breaking down large images! to gain performance on RAM", "white", "on_blue")
        break_start_time = timeit.default_timer()
        if num_workers_breaking <=1:
            for image_url_break_down in tqdm(list_images, desc = "     breaking down big images", colour="yellow"):
                list_images_broke_down = break_down_big_images(image_url_break_down,
                                                    crop_to_foreground = crop_to_foreground,
                                                    treshold_cm = treshold_cm,
                                                    overlap = overlap)
               
        else:
            
            # yazdan.other.pool_execute(yazdan.nnunet.break_down_big_images, list_images, num_workers = num_workers_breaking)
            pool_multi_argument(break_down_big_images, 
                                             zip(list_images, [crop_to_foreground] * len(list_images), [treshold_cm] * len(list_images), [overlap] * len(list_images)), 
                                             num_workers = num_workers_breaking)
        
        break_end_time = timeit.default_timer()
        cprint(f"broke in {np.round((break_end_time - break_start_time)/60, 2)} minutes", "white", "on_yellow")
        if target == "inplace":
            list_images_to_be_ensembeled = [x for x in list_images]
        else:
            list_images_to_be_ensembeled = [os.path.join(target, os.path.basename(x)) for x in list_images]
            
        # list_images_to_be_predicted = [glob(x.replace(".nii.gz", "*")) for x in list_images]
        list_images_to_be_predicted = []
        for url_temp in tqdm(list_images, desc = "Recheck what was inferenced before! ", colour = "cyan"):
            try:
                list_found_temp = glob(url_temp.replace(".nii.gz", "*"))
                list_images_to_be_predicted.append(list_found_temp)
            except:
                cprint("\nError Happened!", "cyan", "on_red")
        list_images_to_be_predicted = flatten_list(list_images_to_be_predicted)
        list_images_to_be_predicted = [x for x in list_images_to_be_predicted if x.endswith("-nnbreak.nii.gz") or x.endswith("-nncrop.nii.gz")]
        list_images = list_images_to_be_predicted
                          
    if organ_names!= "all" and organ_names != ("all",):
        list_models = flatten_list([glob(os.path.join(root_folder, "nnUNet_trained_models", modality,  f"*{x}*")) for x in organ_names])
    else:
        list_models = glob(os.path.join(root_folder, "nnUNet_trained_models", modality, "*"))
     
    if long_image_paths:
        list_images_to_be_ensembeled = [add_long_path_prefix(x) for x in list_images_to_be_ensembeled]
        
    list_models = [x for x in list_models if os.path.isdir(x)]
    for model_folder_url in tqdm(list_models, ncols = 130, colour = "red", desc = "Model Progress"):
        print("\n")
        index = 0
        model_name = os.path.basename(model_folder_url)
        cprint(model_name, "white", "on_yellow")
        model_training_output_dir = glob(os.path.join(model_folder_url, f"*__{config}"))[-1]
        organ_name = model_training_output_dir.split("\\")[-2].split("Dataset")[-1].split("_")[-1]
        # removing what segmented before
        if not overwrite:
            if target == "inplace":
                list_images_pure = [x for x in list_images if not os.path.exists(os.path.join(os.path.dirname(x), os.path.basename(x).replace(".nii.gz", f"--{organ_name}--yzdnn.nii.gz")))]
                list_images_pure = [x for x in list_images_pure if not os.path.exists(os.path.join(os.path.dirname(x), os.path.basename(x).replace("-nncrop.nii.gz", f"--{organ_name}--yzdnn.nii.gz")))]
                # list_images_pure = [x for x in list_images_pure if not os.path.exists(os.path.join(os.path.dirname(x), os.path.basename(x).replace("-nnbreak.nii.gz", f"--{organ_name}--yzdnn.nii.gz")))]
            else:
                list_images_pure = [x for x in list_images if not os.path.exists(os.path.join(target, os.path.basename(x).replace(".nii.gz", f"--{organ_name}--{str(index).zfill(4)}--yzdnn.nii.gz")))]
           
                
        predict_single_model_image_list(list_images_pure,
                model_training_output_dir = model_training_output_dir,
                target = target,
                raw_temp_folder = raw_temp_folder,
                save_probs = save_probs,
                num_worker_preprocessing = num_worker_preprocessing,
                num_worker_saving = num_worker_saving,
                overwrite = overwrite,
                tile_step_size = tile_step_size,
                use_folds = use_folds,
                checkpoint_name = checkpoint_name,
                device = device, 
                clean_before = clean_before,
                clean_after = clean_after,
                perform_everything_on_gpu = perform_everything_on_gpu,
                verbose = verbose,
                use_gaussian = use_gaussian,
                use_mirroring = use_mirroring,
                allow_tqdm = allow_tqdm,
                move_incomplete_segments = move_incomplete_segments,
                sign_images = sign_images,
                )
        time.sleep(5)

        gc.collect()
        torch.cuda.empty_cache()
        
        time.sleep(5)
        
        if ensemble_every_organ_as_completed and break_down_large_images:
            pool_execute(ensemble_images, list_images_to_be_ensembeled, num_workers = num_workers_breaking, desc = "    Ensembling this organ")
            
    if break_down_large_images:
        cprint("\n Getting segments back to original dimensions", "white", "on_cyan")
        pool_multi_argument(ensemble_images,
                                         zip(list_images_to_be_ensembeled, [remove_broken_images_at_end]*len(list_images_to_be_ensembeled), [True] * len(list_images_to_be_ensembeled), [overwrite_ensembling] * len(list_images_to_be_ensembeled)),
                                         num_workers = num_workers_breaking)            
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    return cprint(f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}", "white", "on_blue")            
