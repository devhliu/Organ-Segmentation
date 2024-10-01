def predict_single_model_image_list(list_images,
        model_training_output_dir = r"",
        target = "inplace",
        raw_temp_folder = r"C:\yzdn\nnunet_completed\Raw-Foders-temporary",
        save_probs = False,
        num_worker_preprocessing = 8,
        num_worker_saving = 8,
        overwrite = True,
        tile_step_size = .5,
        use_folds = (0,1,2,3,4),
        checkpoint_name = "checkpoint_final.pth",
        device = "cuda", 
        clean_before = True,
        clean_after = False,
        perform_everything_on_gpu = True,
        verbose = False,
        use_gaussian=True,
        use_mirroring=True,
        allow_tqdm = True,
        move_incomplete_segments = True,
        ):
    import os
    from shutil import copy2 as copyfile
    from shutil import rmtree
    import pandas as pd
    from tqdm import tqdm
    from termcolor import cprint
    from glob import glob
    import time
    # from nnunetv2.paths import nnUNet_results, nnUNet_raw
    import torch
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    # from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    # convert and prepare data
    devic_to_use = torch.device(device, 0)
    use_folds = [x for x in use_folds if os.path.exists(os.path.join(model_training_output_dir, f"fold_{x}"))]
    organ_name = model_training_output_dir.split("\\")[-2].split("Dataset")[-1].split("_")[-1]
    raw_folder_to_segment = os.path.join(raw_temp_folder, organ_name)
    image_temp_folder = os.path.join(raw_folder_to_segment, "images")
    segment_temp_folder = os.path.join(raw_folder_to_segment, "segments")
    
    if move_incomplete_segments:
        cprint("Warninng!!  Moving What have been segmented before !!!! ", "red", "on_cyan")
        try:
            decode_df = pd.read_excel(os.path.join(segment_temp_folder, "decoding-information.xlsx"))
            for row_index in range(len(decode_df)):
                        
                try:
                    copyfile(decode_df.at[row_index, "indexed_segment_url"], decode_df.at[row_index, "final_segment_url"])
                except:
                    # cprint(f"erorr in copying       {decode_df.at[row_index, 'indexed_segment_url']}")
                    pass
        except:
            pass
        

    if not overwrite:
        if target == "inplace":
            list_images = [x for x in list_images if not os.path.exists(os.path.join(os.path.dirname(x), os.path.basename(x).replace(".nii.gz", f"--{organ_name}--yzdnn.nii.gz")))]
        else:
            # this not inplace option is not working now!!!!!!
            list_images = [x for x in list_images if not os.path.exists(os.path.join(target, os.path.basename(x).replace(".nii.gz", f"--{organ_name}--yzdnn.nii.gz")))]
       
    os.makedirs(raw_folder_to_segment, exist_ok=True)
    if clean_before:
        rmtree(raw_folder_to_segment)
    os.makedirs(image_temp_folder, exist_ok=True)
    os.makedirs(segment_temp_folder, exist_ok=True)
    if target != "inplace":
        os.makedirs(target, exist_ok=True)
    decode_df = pd.DataFrame()

    if verbose:
        cprint("Start Decoding Information", "white", "on_cyan")
    
    if len(list_images):
        #filling the dataframe
        # for index, image_url in enumerate(tqdm(list_images, desc = "       Copying Data", colour="blue", ncols = 60)):
        for index, image_url in tqdm(enumerate(list_images), desc = "   Copying files", total = len(list_images), colour = "yellow"):
            image_name = os.path.basename(image_url)
            decode_df.at[index, "image_url"] = image_url
            decode_df.at[index, "image_name"] = image_name
            decode_df.at[index, "index"] = str(index)
            indexed_image_url = os.path.join(image_temp_folder,f"case_{index}_0000.nii.gz" )
            indexed_segment_url = os.path.join(segment_temp_folder,f"case_{index}.nii.gz" )
            decode_df.at[index, "indexed_image_url"] = indexed_image_url
            decode_df.at[index, "indexed_segment_url"] = indexed_segment_url
            if target == "inplace":
                decode_df.at[index, "final_segment_url"] = os.path.join(os.path.dirname(image_url), os.path.basename(image_url).replace(".nii.gz", f"--{organ_name}--yzdnn.nii.gz"))
            else:
                decode_df.at[index, "final_segment_url"] = os.path.join(target, os.path.basename(image_url).replace(".nii.gz", f"--{organ_name}--yzdnn.nii.gz"))
            # copying files
            copyfile(image_url, indexed_image_url)
        decode_df.to_excel(os.path.join(segment_temp_folder, "decoding-information.xlsx"))
        if verbose:
            cprint("Copying files Completed", "white", "on_cyan")
            cprint("Inferences were Started", "white", "on_cyan")
        # instantiate the nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            # perform_everything_on_gpu=perform_everything_on_gpu,
            device=devic_to_use,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=allow_tqdm,
        )
        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir = model_training_output_dir,
            use_folds=use_folds,
            checkpoint_name=checkpoint_name,
        )
        
        predictor.predict_from_files(image_temp_folder,
                                     segment_temp_folder,
                                     save_probabilities=save_probs,
                                     overwrite=True,
                                     num_processes_preprocessing=num_worker_preprocessing, 
                                     num_processes_segmentation_export=num_worker_saving,
                                     folder_with_segs_from_prev_stage=None,
                                     num_parts=1, 
                                     part_id=0)
        
        # yazdan.DL.FreeGPU(close = True, reset = True, gc_collect = True, torch_empty_cache = True)
        # copying the segmentations
        if verbose:
            cprint("Inference was Completed Start Moving Back", "white", "on_cyan")
        for row_index in range(len(decode_df)):
            try:
                copyfile(decode_df.at[row_index, "indexed_segment_url"], decode_df.at[row_index, "final_segment_url"])
                
            except:
                try:
                    copyfile(add_long_path_prefix(decode_df.at[row_index, "indexed_segment_url"]), add_long_path_prefix(decode_df.at[row_index, "final_segment_url"]))
                except:
                    cprint(f"erorr in copying       {decode_df.at[row_index, 'indexed_segment_url']}")
                    pass
        if verbose:
            cprint("Everythin is done", "white", "on_cyan")
        if clean_after:
            rmtree(raw_folder_to_segment)
    else:
        cprint("This model was done before!!  Skipped!")
        
        
        
def is_segment(image, limit = 10):
    import SimpleITK as sitk
    import numpy as np
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    unique_values = np.unique(sitk.GetArrayFromImage(image))
    is_segment = len(unique_values)<limit and all(int(x) == x for x in unique_values)
    return unique_values, is_segment

def pool_multi_argument(map_fuction, arguments, num_workers = 24):
    """
    How to use:
    yazdan.other.pool_multi_argument(map_fuction = yazdan.image.image_segment_slice_show, 
                                     arguments = zip(list_images,list_aorta, list_trget_urls ))
    """
    import multiprocessing
    pool = multiprocessing.Pool(processes = num_workers)
    pool.starmap(map_fuction, arguments)
    pool.close()
    pool.join()
    
def flatten_list(nested_list):
    flattened = []
    for element in nested_list:
        if isinstance(element, list):
            flattened.extend(flatten_list(element))
        else:
            flattened.append(element)
    return flattened
    
def add_long_path_prefix(path):
    # Check if the path already has the extended-length path prefix
    if not path.startswith('\\\\?\\'):
        return f'\\\\?\\{path}'
    return path

def ExtractBetween(string, start, end):
    return (string.split(start))[1].split(end)[0]

def select_objects_larger_than(segment, size_threshold, small_object_replace_value = 0):
    import SimpleITK as sitk
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment, sitk.sitkUInt8)
    labeled_image = sitk.ConnectedComponent(segment)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled_image)
    for label in stats.GetLabels():
        if stats.GetPhysicalSize(label) <= size_threshold:
            # print(stats.GetPhysicalSize(label))
            # Remove objects smaller than or equal to the threshold from the mask
            labeled_image[labeled_image == label] = small_object_replace_value
            # filtered_mask = sitk.BinaryThreshold(labeled_image, lowerThreshold=label + 1, upperThreshold=label + 1, insideValue=0, outsideValue=1)   
    return labeled_image


def keep_largest_segments(segment, engine = "sitk", numbner_of_objects = 1):
    import SimpleITK as sitk
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
        segment = sitk.Cast(segment, sitk.sitkUInt8)
    if engine == "sitk":
        component_image = sitk.ConnectedComponent(segment)
        sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
        # largest_component_binary_image = sorted_component_image == 1
        largest_component_binary_image = sum([sorted_component_image == label for label in range(1, numbner_of_objects + 1)])
    elif engine == "monai":
        from monai.transforms import KeepLargestConnectedComponent
        segment_array = sitk.GetArrayFromImage(segment)
        keep_large_transform = KeepLargestConnectedComponent(num_components = numbner_of_objects)
        largest_component_binary_array = keep_large_transform(segment_array)
        largest_component_binary_image = sitk.GetImageFromArray(largest_component_binary_array)
    return largest_component_binary_image



def body_segment(url, lower_hu = -300, metal_hu = 2000, object_min_size = 10, close_voxels = 1, keep_largest = True):
    import SimpleITK as sitk
    if isinstance(url, str):
        image = sitk.ReadImage(url)
    elif isinstance(url, sitk.Image):
        image = url
    image = sitk.Cast(image, sitk.sitkFloat32)
    bcct = sitk.BinaryThreshold(image, lowerThreshold=lower_hu, upperThreshold=metal_hu, insideValue=1, outsideValue=0)
    bcct = sitk.BinaryMorphologicalClosing(bcct, [close_voxels, close_voxels, close_voxels])  # Closing operation to fill small gaps
    for slice in range(bcct.GetSize()[2]):
        bcct[:,:,slice] = sitk.BinaryFillhole(bcct[:,:,slice], fullyConnected = True)  # Fill any remaining holes inside the body
    bcct = sitk.ConnectedComponent(bcct)
    
    bcct = select_objects_larger_than(bcct, object_min_size)
    if keep_largest:
        bcct = keep_largest_segments(bcct)
    bcct = bcct > 0
    bcct = sitk.Cast(bcct, sitk.sitkUInt8)
    bed_no_use = bcct
    return bcct, bed_no_use

def pool_execute(map_function, list_inputs, num_workers = 24, desc = "Doing task in parallel"):
    import multiprocessing
    from tqdm import tqdm
    pool = multiprocessing.Pool(maxtasksperchild = num_workers, processes = num_workers)
    mapped_values = list(tqdm(pool.imap_unordered(map_function, list_inputs), total=len(list_inputs), colour = "blue", desc = desc))
    return mapped_values
    
def match_space(input_image, reference_image, interpolate = "linear", DefaultPixelValue = 0):
    import SimpleITK as sitk
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image)
    if isinstance(reference_image, str):
        reference_image = sitk.ReadImage(reference_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_image.GetSpacing())
  
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    # Set the default pixel value to -1000
    resampler.SetDefaultPixelValue(DefaultPixelValue)
    if interpolate == "linear":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolate == "nearest":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolate.lower() == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resampler.Execute(input_image)
    return resampled_image

  
def crop_image_to_segment(image, segment, crop_dims = "all", margin_mm = 0, 
                          lowerThreshold = 0.1, upperThreshold = .9,
                          insideValue = 0, outsideValue =1, 
                          force_match = False
                          ):
    import SimpleITK as sitk
    from termcolor import cprint
    if isinstance(image, str):
        image = sitk.ReadImage(image)
        image = sitk.DICOMOrient(image, "LPS")
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
        segment = sitk.DICOMOrient(segment, "LPS")
        # finding crop area
    if force_match:
        segment = match_space(input_image = segment, reference_image = image)
    segment = sitk.Cast(segment, sitk.sitkUInt8)
    segment_non_binary = segment
    segment = sitk.BinaryThreshold(segment, lowerThreshold=lowerThreshold, 
                                   upperThreshold=upperThreshold, 
                                   insideValue = insideValue, outsideValue = outsideValue)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(segment)
    bounding_box = label_shape_filter.GetBoundingBox(1) 

    start_physical_point = segment.TransformIndexToPhysicalPoint(bounding_box[0 : int(len(bounding_box) / 2)])
    end_physical_point = segment.TransformIndexToPhysicalPoint([x+sz for x,sz in zip(bounding_box[0 : int(len(bounding_box) / 2)], bounding_box[int(len(bounding_box) / 2) :])])
    if any([start>end for start, end in zip(start_physical_point, end_physical_point)]):
        cprint("warning directions have issues, check the output !!!!!", "white", "on_red")
    
    start_physical_point = [x - margin_mm for x in start_physical_point]
    end_physical_point = [x + margin_mm for x in end_physical_point]
    # crop using the indexes
    image_crop_start_indices = image.TransformPhysicalPointToIndex(start_physical_point)
    image_crop_end_indices = image.TransformPhysicalPointToIndex(end_physical_point)
    
    segment_crop_start_indices = segment.TransformPhysicalPointToIndex(start_physical_point)
    segment_crop_end_indices = segment.TransformPhysicalPointToIndex(end_physical_point)
    
    
    image_crop_sizes = [a-b for a,b in zip(image_crop_end_indices , image_crop_start_indices)]
    segment_crop_sizes = [a-b for a,b in zip(segment_crop_end_indices , segment_crop_start_indices)]
            
    
    image_crop_start_indices = list(image_crop_start_indices)
    for dimension, image_crop_start_index in enumerate(image_crop_start_indices):
        if image_crop_start_index < 0 : 
            image_crop_start_indices[dimension] = 0

    image_crop_sizes = list(image_crop_sizes)
    for dimension, image_crop_size in enumerate(image_crop_sizes):
        if image_crop_size + image_crop_start_indices[dimension] > image.GetSize()[dimension]: 
            image_crop_sizes[dimension] = image.GetSize()[dimension] - image_crop_start_indices[dimension] -1
        
    segment_crop_start_indices = list(segment_crop_start_indices)
    for dimension, segment_crop_start_index in enumerate(segment_crop_start_indices):
        if segment_crop_start_index < 0 : 
            segment_crop_start_indices[dimension] = 0


    segment_crop_sizes = list(segment_crop_sizes)
    for dimension, segment_crop_size in enumerate(segment_crop_sizes):
        if segment_crop_size + segment_crop_start_indices[dimension] > segment.GetSize()[dimension]: 
            segment_crop_sizes[dimension] = segment.GetSize()[dimension] - segment_crop_start_indices[dimension] -1
            
    image_crop_start_indices = list(image_crop_start_indices)
    if crop_dims == "all":
        "do nothging -- crop in all dimension"
    else:
        no_crop_dims = [x for x in [0,1,2] if x not in crop_dims]
        for dimension in no_crop_dims:
            image_crop_start_indices[dimension] = 0
            image_crop_sizes[dimension] = image.GetSize()[dimension]
    
    image_cropped = sitk.RegionOfInterest(image, image_crop_sizes, image_crop_start_indices)
    segment_cropped = sitk.RegionOfInterest(segment, segment_crop_sizes, segment_crop_start_indices)
    segment_non_binary_cropped = sitk.RegionOfInterest(segment_non_binary, segment_crop_sizes, segment_crop_start_indices)
    crop_box_out = {}
    crop_box_out["start_physical_point"] = start_physical_point
    crop_box_out["end_physical_point"] = end_physical_point
    crop_box_out["crop_start_indices"] = image_crop_start_indices
    crop_box_out["crop_end_indices"] = image_crop_end_indices
    crop_box_out["crop_sizes"] = image_crop_sizes
    
    return image_cropped, segment_cropped, segment_non_binary_cropped, crop_box_out 


def ensemble_images(image_url, remove_broken_images = False, remove_broken_segments = False, overWrite = False):
    """ this is working well with images broke wth overlap == 0"""
    import SimpleITK as sitk
    from glob import glob
    import os
    from natsort import os_sorted
    from termcolor import cprint
    image_name = os.path.basename(image_url).replace(".nii.gz", "")
    image_folder = os.path.dirname(image_url)
    list_parts_images = os_sorted(glob(os.path.join(image_folder, f"{image_name}-part*-nnbreak.nii.gz")))
    crop_image_url = image_url.replace(".nii.gz", "-nncrop.nii.gz")
    list_croped_segments = os_sorted(glob(crop_image_url.replace(".nii.gz", "*")))
    list_croped_segments = [x for x in list_croped_segments if "--yzdnn.nii.gz" in x]
    for cropped_segment in list_croped_segments:
        segment_original_url = cropped_segment.replace("-nncrop", "")
        if not os.path.exists(segment_original_url):
            segment_original = match_space(input_image = cropped_segment, reference_image = image_url)
            sitk.WriteImage(segment_original, segment_original_url)
        if remove_broken_segments:
            try:
                os.remove(cropped_segment)
            except:
                pass
    if remove_broken_images:
        try:
            os.remove(crop_image_url)
        except:
            pass
            
    list_parts_all = os_sorted(glob(os.path.join(image_folder, f"{image_name}-part*-nnbreak--*--yzdnn.nii.gz")))
    
    list_organs = set([ExtractBetween(x, "-nnbreak--", "--yzdnn.nii.gz") for x in list_parts_all])
    for organ_name in list_organs:
        list_parts = os_sorted(glob(os.path.join(image_folder, f"{image_name}-part*-nnbreak-*{organ_name}*--yzdnn.nii.gz")))
        if not os.path.exists(image_url.replace(".nii.gz", f"--{organ_name}--yzdnn.nii.gz")) or overWrite:
            segment_null = sitk.ReadImage(image_url)
            segment_null[:,:,:] = 0
            segment_null = sitk.Cast(segment_null, sitk.sitkUInt8) 
            for part_url in list_parts:
                image_break_temp = match_space(input_image = part_url,
                                                            reference_image = segment_null)
                image_break_temp = sitk.Cast(image_break_temp, sitk.sitkUInt8) 
                segment_null = segment_null + image_break_temp
            segment_null = sitk.Cast(segment_null, sitk.sitkUInt8)   
            sitk.WriteImage(segment_null, image_url.replace(".nii.gz", f"--{organ_name}--yzdnn.nii.gz"))
        if remove_broken_segments:
            try:
                [os.remove(x) for x in list_parts]
            except:
                pass
    if remove_broken_images:
        try:
            [os.remove(x) for x in list_parts_images]
        except:
            pass
        
     
def CopyInfo(ReferenceImage, UpdatingImage, origin = True, spacing = True, direction = True):
    import SimpleITK as sitk
    if isinstance(ReferenceImage, str):
        ReferenceImage = sitk.ReadImage(ReferenceImage)
    if isinstance(UpdatingImage, str):
        UpdatingImage = sitk.ReadImage(UpdatingImage)
    UpdatedImage = UpdatingImage 
    if origin:
        UpdatedImage.SetOrigin(ReferenceImage.GetOrigin())
    if spacing:
        UpdatedImage.SetSpacing(ReferenceImage.GetSpacing())
    if direction:
        UpdatedImage.SetDirection(ReferenceImage.GetDirection())
    return UpdatedImage


def select_segment_number(segment, segment_value):
    import SimpleITK as sitk
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
    segment_array = sitk.GetArrayFromImage(segment)
    segment_array[segment_array != segment_value] = 0
    segment_array[segment_array == segment_value] = 1
    segment_single = sitk.GetImageFromArray(segment_array)
    segment_single = CopyInfo(segment, segment_single)
    return segment_single, segment_array

def segment_volume(segment, spacing = "from-segment", segment_number = 1):
    """
    Parameters
    ----------
    segment : segment URL or segment sitk image.
    spacing : TYPE, optional
        DESCRIPTION. The default is "from-segment".
    segment_number : TYPE, optional
        DESCRIPTION. The default is 1.
    Returns
    -------
    number_of_voxels : number of voxels in the semgnet
    segment_volume : segment volume in ml (1e3 mm3)
    """
    import SimpleITK as sitk
    import numpy as np
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
    if spacing == "from-segment":
        spacing = segment.GetSpacing()
  
    segment_binary = select_segment_number(segment, segment_value = segment_number)[0]
    segment_array = sitk.GetArrayFromImage(segment_binary)
    number_of_voxels =  segment_array.sum()
    segment_volume = number_of_voxels * np.prod(list(spacing)) / 1e3
    return number_of_voxels, segment_volume



def break_down_big_images(image_url, crop_to_foreground = True, treshold_cm = 30, overlap = 0, ignore_less_than_cm = 2):
    import SimpleITK as sitk
    import numpy as np
    import os
    from termcolor import cprint
    try:
        if isinstance(image_url, str):
            image = sitk.ReadImage(image_url)
        else: 
            image = image_url
        cprint(".", "white", "on_yellow", end = "")
        image_size = image.GetSize()
        pix_dim = image.GetSpacing()
        scan_length = image_size[2] * pix_dim[2] * .1 # this  is in cm
        number_of_parts = int(np.ceil(scan_length / treshold_cm))
        sample_size = round(image_size[2] / number_of_parts)
        list_images_broke_down = []
        if scan_length < ignore_less_than_cm:
            cprint("image is too small", "white", "on_yellow")
            return list_images_broke_down
        if scan_length > treshold_cm:
            for part in range(number_of_parts):
                # locals()[f"image_part_{i}"] 
                broke_down_temp_url = image_url.replace(".nii.gz", f"-part-{part}-nnbreak.nii.gz")
                if not os.path.exists(broke_down_temp_url):
                    locals()[f"image_part_{part}"] = image[:,:, part * sample_size: ((part + 1) * sample_size) + overlap]
                    if crop_to_foreground:
                        body_contour = image.body_segment(locals()[f"image_part_{part}"])[0]
                        if segment_volume(body_contour)[-1] < 10:
                            pass
                            # body_contour = sitk.Image(body_contour.GetSize(), sitk.sitkUInt8) + 1
                            # yazdan.image.view([locals()[f"image_part_{part}"], body_contour], pause_code=True)
                            # locals()[f"image_part_{part}"] = locals()[f"image_part_{part}"]
                        else:
                            locals()[f"image_part_{part}"] = crop_image_to_segment(locals()[f"image_part_{part}"], body_contour)[0]
                    if isinstance(image_url, str):
                        sitk.WriteImage(locals()[f"image_part_{part}"], image_url.replace(".nii.gz", f"-part-{part}-nnbreak.nii.gz"))
                        list_images_broke_down.append(image_url.replace(".nii.gz", f"-part-{part}-nnbreak.nii.gz"))
                else:
                    list_images_broke_down.append(image_url.replace(".nii.gz", f"-part-{part}-nnbreak.nii.gz"))

        else:
            if not os.path.exists(image_url.replace(".nii.gz", "-nncrop.nii.gz")):
                body_contour = body_segment(image)[0]
                image_cropped = crop_image_to_segment(image, body_contour)[0]
                if isinstance(image_url, str):
                    sitk.WriteImage(image_cropped, image_url.replace(".nii.gz", "-nncrop.nii.gz"))
                    list_images_broke_down.append(image_url.replace(".nii.gz", "-nncrop.nii.gz"))
    except:
        list_images_broke_down = []
        cprint(f"this image was not broke! \n {image_url}", "white", "on_red")
    return list_images_broke_down
