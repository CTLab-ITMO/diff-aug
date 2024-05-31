import json
from pathlib import Path


PIPELINE_PATH = 'workflows/inpainting_SDXL_v5' + \
    '_release_alpha_0.76_small_fixes_api.json'


def modificate_workflow_v5(image_path: str, mask_path: str,
                           data_images_path: str, data_masks_path: str,
                           output_path: str, output_name: str,
                           main_canny_weight: float, main_depth_weight: float,
                           main_soft_edge_weight: float,
                           main_usual_ipadapter_weight: float,
                           main_plus_ipadapter_weight: float,
                           main_neg_plus_ipadapter_weight: float,
                           dataset_name: str, positive_prompt: str,
                           negative_prompt: str, seed: int) -> str:
    """Function to modify the ComfyUI workflow v5 with the given parameters.

    Args:
        image_path (str): full path to an image to inpaint
        mask_path (str): full path to a black-white mask of the image
        data_images_path (str): full path to dataset images
        data_masks_path (str): full path to dataset masks
        output_path (str): path to folder where the inpainted image
            will be saved
        output_name (str): name of the inpainted image without extension
        main_canny_weight (float): Canny Edge ControlNet weight for
            the main model
        main_depth_weight (float): Depth ControlNet weight for the main model
        main_soft_edge_weight (float): Soft Edge ControlNet weight for
            the main model
        main_usual_ipadapter_weight (float): IPAdapter weight for
            the main model
        main_plus_ipadapter_weight (float): IPAdapter (plus) weight for
            the main model
        main_neg_plus_ipadapter_weight (float): IPAdapter (negative plus)
            weight for the main model
        dataset_name (str): Database name (needed for CLIP embeddings storage)
        positive_prompt (str): Positive prompt
        negative_prompt (str): Negative prompt
        seed (int): Generation seed

    Returns:
        str: JSON string with the modified pipeline
    """

    assert Path(image_path).exists(), \
        f'Path {image_path} does not exist'
    assert Path(mask_path).exists(), \
        f'Path {mask_path} does not exist'
    assert Path(data_images_path).exists(), \
        f'Path {data_images_path} does not exist'
    assert Path(data_masks_path).exists(), \
        f'Path {data_masks_path} does not exist'
    assert Path(output_path).exists(), \
        f'Path {output_path} does not exist'

    with open(PIPELINE_PATH) as f:
        pipeline = json.load(f)

    # controlnet weights "canny", "depth", "soft-edge"
    pipeline['327']['inputs']['strength'] = main_canny_weight
    pipeline['142']['inputs']['strength'] = main_depth_weight
    pipeline['4']['inputs']['strength'] = main_soft_edge_weight

    # ipadapter weights "usual", "plus", "negative plus"
    pipeline['30']['inputs']['weight'] = main_usual_ipadapter_weight
    pipeline['141']['inputs']['weight'] = main_plus_ipadapter_weight
    pipeline['61']['inputs']['weight'] = main_neg_plus_ipadapter_weight

    # set images loader path
    pipeline['311']['inputs']['Text'] = data_images_path
    # set masks loader path
    pipeline['312']['inputs']['Text'] = data_masks_path
    # set inpainting images saver path
    pipeline['184']['inputs']['path'] = output_path

    # database name
    pipeline['313']['inputs']['Text'] = dataset_name

    # generation seed
    pipeline['166']['inputs']['seed'] = seed

    # set positive prompt
    pipeline['167']['inputs']['text'] = positive_prompt
    # set negative prompt
    pipeline['168']['inputs']['text'] = negative_prompt

    pipeline['309']['inputs']['Text'] = image_path
    pipeline['310']['inputs']['Text'] = mask_path
    pipeline['184']['inputs']['filename'] = output_name

    return json.dumps(pipeline)
