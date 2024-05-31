from pathlib import Path
from .workflow import modificate_workflow_v5
import numpy as np
from comfy_runner.inf import ComfyRunner


def run_augmentation(data_images_path: str, data_masks_path: str,
                     output_path: str,
                     number_of_inpainted_images_per_image_required: int,
                     main_canny_weight: float, main_depth_weight: float,
                     main_soft_edge_weight: float,
                     main_usual_ipadapter_weight: float,
                     main_plus_ipadapter_weight: float,
                     main_neg_plus_ipadapter_weight: float,
                     dataset_name: str, positive_prompt: str,
                     negative_prompt: str, seed: int) -> None:
    """Function to run the augmentation loop with the given parameters.

    Args:
        data_images_path (str): full path to dataset images
        data_masks_path (str): full path to dataset masks
        output_path (str): path to folder where the inpainted image
            will be saved
        number_of_inpainted_images_per_image_required (int): number of
            inpainted images per image required
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
    """

    assert Path(data_images_path).exists(), \
        f'Path {data_images_path} does not exist'
    assert Path(data_masks_path).exists(), \
        f'Path {data_masks_path} does not exist'
    assert Path(output_path).exists(), \
        f'Path {output_path} does not exist'

    data_images_path = Path(data_images_path)
    image_list = (list(data_images_path.glob('*.jpg')) +
                  list(data_images_path.glob('*.png')) +
                  list(data_images_path.glob('*.jpeg')))

    runner = ComfyRunner()

    for img_idx, image_path in enumerate(image_list):
        for inp_idx in range(number_of_inpainted_images_per_image_required):
            output_name = f'{image_path.stem}-{str(inp_idx)}'
            mask_path = str(Path(data_masks_path) / image_path.stem) + '.png'

            np.random.seed(seed + (inp_idx + 1) * (img_idx + 1))
            gen_seed = int(np.random.randint(1, 4294967294, dtype=np.int64))

            pipeline = modificate_workflow_v5(str(image_path), mask_path,
                                              data_images_path,
                                              data_masks_path,
                                              output_path, output_name,
                                              main_canny_weight,
                                              main_depth_weight,
                                              main_soft_edge_weight,
                                              main_usual_ipadapter_weight,
                                              main_plus_ipadapter_weight,
                                              main_neg_plus_ipadapter_weight,
                                              dataset_name, positive_prompt,
                                              negative_prompt, gen_seed)

            runner.predict(
                workflow_input=pipeline,
                stop_server_after_completion=False
            )

    runner.stop_server()
