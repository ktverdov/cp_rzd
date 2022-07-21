python3 ./tools/train_pl.py --config "./configs/pl/unet_tfefficientnetb3_hard512+896_cedice.yaml" --fold 0
python3 ./tools/train_pl.py --config "./configs/pl/unet_tfefficientnetb3_hard512+896_cedice.yaml" --fold 1
python3 ./tools/train_pl.py --config "./configs/pl/unet_tfefficientnetb3_hard512+896_cedice.yaml" --fold 2
python3 ./tools/train_pl.py --config "./configs/pl/unet_tfefficientnetb3_hard512+896_cedice.yaml" --fold 3
python3 ./tools/train_pl.py --config "./configs/pl/unet_tfefficientnetb3_hard512+896_cedice.yaml" --fold 4

python3 ./tools/train_pl.py --config "./configs/pl/fpn_hrnetw48_semihard512+896_cedice_accumulate8.yaml" --fold 0
python3 ./tools/train_pl.py --config "./configs/pl/fpn_hrnetw48_semihard512+896_cedice_accumulate8.yaml" --fold 1
python3 ./tools/train_pl.py --config "./configs/pl/fpn_hrnetw48_semihard512+896_cedice_accumulate8.yaml" --fold 2
python3 ./tools/train_pl.py --config "./configs/pl/fpn_hrnetw48_semihard512+896_cedice_accumulate8.yaml" --fold 3
python3 ./tools/train_pl.py --config "./configs/pl/fpn_hrnetw48_semihard512+896_cedice_accumulate8.yaml" --fold 4
