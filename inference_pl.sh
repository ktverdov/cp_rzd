python3 ./tools/inference_pl.py --configs "./configs/pl/unet_tfefficientnetb3_hard512+896_cedice.yaml" \
    "./configs/pl/fpn_hrnetw48_semihard512+896_cedice_accumulate8.yaml" \
    --weights 0.35 0.65 \
    --test_dir "./data/test/" \
    --output_dir "./results/" \
    --folds_ensemble \
    --tta \
    --resize_first 
