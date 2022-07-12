import os
import pandas as pd
import cv2
from collections import defaultdict
 
from torch.utils.data import Dataset
 
from .utils import load_yaml, load_augs
 

def create_datasets(config, fold):
    datasets = defaultdict(dict)
    
    datasets["train"] = create_dataset(config.train_data, "train", fold)
    datasets["val"] = create_dataset(config.val_data, "val", fold)

    return datasets


def create_dataset(config, mode, fold):
    if config.type == "SegmDataset":
        num_classes = 4 if "num_classes" not in config else config.num_classes
        dataset = SegmDataset(config.info_path, 
                              mode, 
                              config.augs_config_path, 
                              fold, num_classes)
    else:
        dataset = None
    
    return dataset


class SegmDataset(Dataset):
    def __init__(self, data_path, mode, augs_config_path, fold, num_classes):
        data = pd.read_csv(data_path)

        if mode == "train":
            data = data[data["fold"] != fold]
        else:
            data = data[data["fold"] == fold]

        self.images_paths = data["image_path"].values
        self.masks_paths = data["mask_path"].values

        augs_config = load_yaml(augs_config_path)
        self.augs = load_augs(augs_config)

        self.num_classes = num_classes

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        mask_path = self.masks_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        augmented = self.augs(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        return image, mask
    
    def get_num_classes(self):
        return self.num_classes
