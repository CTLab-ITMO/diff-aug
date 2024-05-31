call ./venv/bin/activate.bat
set data_images_path="/path/to/data/images"
set data_masks_path="/path/to/data/masks"
set output_path="/path/to/output/folder"
set number_of_inpainted_images_per_image_required=1
set main_canny_weight=0.5
set main_depth_weight=0.5
set main_soft_edge_weight=0.5
set main_usual_ipadapter_weight=0.5
set main_plus_ipadapter_weight=0.5
set main_neg_plus_ipadapter_weight=0.5
set dataset_name="dataset_name"
set positive_prompt="positive_prompt"
set negative_prompt="negative_prompt"
set seed=42

python main.py --data_images_path %data_images_path%^
 --data_masks_path %data_masks_path%^
  --output_path %output_path%^
   --number_of_inpainted_images_per_image_required %number_of_inpainted_images_per_image_required%^
    --main_canny_weight %main_canny_weight%^
     --main_depth_weight %main_depth_weight%^
      --main_soft_edge_weight %main_soft_edge_weight%^
       --main_usual_ipadapter_weight %main_usual_ipadapter_weight%^
        --main_plus_ipadapter_weight %main_plus_ipadapter_weight%^
         --main_neg_plus_ipadapter_weight %main_neg_plus_ipadapter_weight%^
          --dataset_name %dataset_name%^
           --positive_prompt %positive_prompt%^
            --negative_prompt %negative_prompt%^
             --seed %seed%
