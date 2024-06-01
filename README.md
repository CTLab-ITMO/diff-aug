# Diff-Aug: Augmentation method based on diffusion models for object detection and segmentation

## Requirements:

1. Install pip.
2. Install Cuda Toolkit 12.1

## Installation (Windows):

```
setup.bat
```

## Installation (Linux):

To be done.

## Usage:

On Windows, you can run the following command to start the augmentation process:

```
run.bat
```

Before running the script, you need to modify the `run.bat` file to specify the augmentation parameters:
1. `data_images_path` - path to the folder with images.
2. `data_masks_path` - path to the folder with masks.
3. `output_path` - path to the output folder.
4. `number_of_inpainted_images_per_image_required` - number of augmented images per image required.
5. `main_canny_weight` - weight of the canny ControlNet for the main model.
6. `main_depth_weight` - weight of the depth ControlNet for the main model.
7. `main_soft_edge_weight` - weight of the soft edge ControlNet for the main model.
8. `main_usual_ipadapter_weight` - weight of the IPAdapter for general features of neighboring images for the main model.
9. `main_plus_ipadapter_weight` - weight of the IPAdapter (Plus) for input image features for the main model.
10. `main_neg_plus_ipadapter_weight` - weight of the IPAdapter (Plus) for negative object features of neighboring images for the main model.
11. `dataset_name` - name of the dataset for CLIP features storage.
12. `positive_prompt` - positive generation prompt.
13. `negative_prompt` - negative generation prompt.
14. `seed` - random generation seed.

Alternatively, you can run the augmentation process via Python script:

```
from src.aug_loop import run_augmentation

run_augmentation = run_augmentation(
    ...
)
```
