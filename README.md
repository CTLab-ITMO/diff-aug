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

```python
from src.aug_loop import run_augmentation

run_augmentation(
    ...
)
```

## Method schema:

![image](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/5fc3d1e9-d417-492c-abfa-07fca36a434a)

![image](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/5e2ef05a-d858-418c-ae57-a061aced008a)

![image](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/fc234f32-dbde-424b-b868-f852f6ea5230)

![image](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/f2e6a0dd-436d-4c5c-a1f9-91bcf13c6200)

![image](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/b4ffbd2f-59a7-4394-9d4c-b1d98fe206e5)

## Examples:
Generation examples on the [Potholes](https://universe.roboflow.com/final-project-iic7d/pothole-detection-system-new/dataset/1) dataset:

![image](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/4d307bbd-bb97-42db-aec0-c66596ddd330)

Generation examples on the [Rooftops](https://universe.roboflow.com/snowcity/roof-jwa0b/dataset/10) dataset:

![image](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/3259e468-a60a-446e-9850-f307138f5b2a)

# Diff-Aug: 	Аугментация изображений для задач детекции и сегментации на основе диффузионных нейронных сетей

## Требования

1. Установите pip.
2. Установите Cuda Toolkit 12.1

## Установка (Windows):

```
setup.bat
```

## Установка (Linux):

В процессе.

## Использование:   

На Windows вы можете запустить следующую команду, чтобы начать процесс аугментации:

```
run.bat
```

Перед запуском скрипта вам необходимо изменить файл `run.bat`, чтобы указать параметры аугментации:
1. `data_images_path` - путь к папке с изображениями.
2. `data_masks_path` - путь к папке с масками.
3. `output_path` - путь к папке вывода.
4. `number_of_inpainted_images_per_image_required` - количество аугментированных изображений на одно изображение.
5. `main_canny_weight` - вес Canny ControlNet для основной модели.
6. `main_depth_weight` - вес Depth ControlNet для основной модели.
7. `main_soft_edge_weight` - вес Soft Edge ControlNet для основной модели.
8. `main_usual_ipadapter_weight` - вес IPAdapter для общих признаков соседних изображений для основной модели.
9. `main_plus_ipadapter_weight` - вес IPAdapter (Plus) для признаков входного изображения для основной модели.
10. `main_neg_plus_ipadapter_weight` - вес IPAdapter (Plus) для отрицательных признаков объектов соседних изображений для основной модели.
11. `dataset_name` - имя набора данных для хранения признаков CLIP.
12. `positive_prompt` - положительный промпт генерации.
13. `negative_prompt` - отрицательный промпт генерации.
14. `seed` - случайное зерно генерации.

Кроме того, вы можете запустить процесс аугментации через скрипт Python:

```python
from src.aug_loop import run_augmentation

run_augmentation(
    ...
)
```

## Схема метода:

![my_method_v3](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/f2cedf20-0175-4b8e-a960-9e22d811c3f8)

![my_method_v3_target_objects](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/41fd8b68-3f55-476e-be44-fd1b1feab988)

![my_method_v3_features_1](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/aa2ee428-436c-466f-9827-a161ccff02cd)

![my_method_v3_features_2](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/b813f5a1-4744-4103-b1c9-0ef718c5a437)

![my_method_v3_controlnets](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/ba9535b1-e668-4174-8aa2-aad36d70bf42)

## Примеры:

Примеры генерации на датасете [Potholes](https://universe.roboflow.com/final-project-iic7d/pothole-detection-system-new/dataset/1):

![image](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/7eba75d1-f314-40cd-bd08-6782fea89fe7)

Примеры генерации на датасете [Rooftops](https://universe.roboflow.com/snowcity/roof-jwa0b/dataset/10):

![image](https://github.com/CTLab-ITMO/diff-aug/assets/29786176/4781207b-5e04-4246-ad9a-1e97c6885d69)