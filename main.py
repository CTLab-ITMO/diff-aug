import argparse
from src.aug_loop import run_augmentation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_images_path', type=str, required=True)
    parser.add_argument('--data_masks_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--number_of_inpainted_images_per_image_required',
                        type=int, required=True)
    parser.add_argument('--main_canny_weight', type=float, required=True)
    parser.add_argument('--main_depth_weight', type=float, required=True)
    parser.add_argument('--main_soft_edge_weight', type=float, required=True)
    parser.add_argument('--main_usual_ipadapter_weight', type=float,
                        required=True)
    parser.add_argument('--main_plus_ipadapter_weight', type=float,
                        required=True)
    parser.add_argument('--main_neg_plus_ipadapter_weight', type=float,
                        required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--positive_prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    run_augmentation = run_augmentation(
        data_images_path=args.data_images_path,
        data_masks_path=args.data_masks_path,
        output_path=args.output_path,
        number_of_inpainted_images_per_image_required=args.number_of_inpainted_images_per_image_required,  # noqa
        main_canny_weight=args.main_canny_weight,
        main_depth_weight=args.main_depth_weight,
        main_soft_edge_weight=args.main_soft_edge_weight,
        main_usual_ipadapter_weight=args.main_usual_ipadapter_weight,
        main_plus_ipadapter_weight=args.main_plus_ipadapter_weight,
        main_neg_plus_ipadapter_weight=args.main_neg_plus_ipadapter_weight,
        dataset_name=args.dataset_name,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed
    )
