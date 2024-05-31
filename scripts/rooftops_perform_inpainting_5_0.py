import json
from pathlib import Path
from urllib import request
import numpy as np


def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)


pipeline_path = 'inpainting_SDXL_v5_release_alpha_0.9_hyper_and_upscale_api.json'
images_paths = ['C:\\Users\\PnthrLeo\\Desktop\\yolo-rooftops\\data\\rooftops\\rooftops-orig\\train\\images']
masks_paths = ['C:\\Users\\PnthrLeo\\Desktop\\yolo-rooftops\\data\\rooftops\\rooftops-orig\\train\\masks']
inpainting_paths = ['C:\\Users\\PnthrLeo\\Desktop\\yolo-rooftops\\data\\rooftops\\rooftops-inpainting\\train-v5\\images']

# verify that paths exist
for path in images_paths:
    assert Path(path).exists(), f'Path {path} does not exist'
for path in masks_paths:
    assert Path(path).exists(), f'Path {path} does not exist'
for path in inpainting_paths:
    assert Path(path).exists(), f'Path {path} does not exist'

for part_idx in range(len(images_paths)):
    number_of_inpainted_images_per_image_required = 6

    with open(f'../comfyUI/workflows/{pipeline_path}') as f:
        pipeline = json.load(f)

    # # controlnet weights "canny", "depth", "soft-edge" 
    # pipeline['327']['inputs']['strength'] = 0.7
    # pipeline['142']['inputs']['strength'] = 0.2
    # pipeline['4']['inputs']['strength'] = 0.2

    # # ipadapter weights "usual", "plus", "negative plus"
    # pipeline['30']['inputs']['weight'] = 0.1
    # pipeline['141']['inputs']['weight'] = 0.5
    # pipeline['61']['inputs']['weight'] = -0.3

    # # masking settings
    # pipeline['134']['inputs']['expand'] = -4
    # pipeline['289']['inputs']['expand'] = 3

    # set images loader path
    pipeline['311']['inputs']['Text'] = images_paths[part_idx]
    # set masks loader path
    pipeline['312']['inputs']['Text'] = masks_paths[part_idx]
    # set inpainting images saver path
    pipeline['184']['inputs']['path'] = inpainting_paths[part_idx]
    pipeline['184']['inputs']['extension'] = 'jpeg'
    # database name
    pipeline['313']['inputs']['Text'] = 'rooftops'

    positive_prompt = ''
    negative_prompt = ''

    for img_idx, image_path in enumerate(list(Path(images_paths[part_idx]).glob('*.jpg'))):
        inpainted_images_counter = 0

        while inpainted_images_counter < number_of_inpainted_images_per_image_required:
            output_name = f'{image_path.stem}-{str(inpainted_images_counter)}-{str(0)}'
            if (Path(inpainting_paths[part_idx]) / (output_name + '.jpeg')).exists():
                inpainted_images_counter += 1
                continue
            # set seed
            np.random.seed(inpainted_images_counter * (img_idx + 1))
            pipeline['166']['inputs']['seed'] = int(np.random.randint(1, 4294967294, dtype=np.int64))

            # set positive prompt
            pipeline['167']['inputs']['text'] = positive_prompt
            # set negative prompt
            pipeline['168']['inputs']['text'] = negative_prompt

            pipeline['309']['inputs']['Text'] = str(image_path)
            pipeline['310']['inputs']['Text'] = str(Path(masks_paths[part_idx]) / image_path.stem) + '.png'
            pipeline['184']['inputs']['filename'] = output_name
            queue_prompt(pipeline)

            inpainted_images_counter += 1
