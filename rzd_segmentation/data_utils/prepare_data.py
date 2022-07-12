import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold


class2internal_mapping = {0: 0, 6: 1,  7: 2, 10: 3}


def convert_mask(input_dir, path, output_dir):
    mask = cv2.imread(os.path.join(input_dir, path), cv2.IMREAD_GRAYSCALE)
    mask_remaped = mask.copy()
    
    for old, new in class2internal_mapping.items():
        mask_remaped[mask == old] = new

    cv2.imwrite(os.path.join(output_dir, path), mask_remaped)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--masks_dir', required=True)
    parser.add_argument('--new_masks_dir', required=True)
    parser.add_argument('--output_df_path', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.new_masks_dir):
        os.mkdir(args.new_masks_dir)
    
    for mask_name in tqdm(os.listdir(args.masks_dir)):
        convert_mask(args.masks_dir, mask_name, args.new_masks_dir)


    images_names = os.listdir(args.images_dir)
    data = pd.DataFrame({"image_name": images_names})
    data["image_path"] = args.images_dir + data["image_name"]
    data["mask_path"] = args.new_masks_dir + data["image_name"]
    data["fold"] = -1

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (_, test_index) in enumerate(kf.split(data)):
        data.loc[test_index, "fold"] = fold

    data.to_csv(args.output_df_path, index=False)
