import os
import shutil
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import glob
import ttach as tta
import skimage
from tqdm import tqdm


from rzd_segmentation.pl_training.models import create_model, load_model_weights
from rzd_segmentation.pl_training.utils import load_config_file
from rzd_segmentation.pl_training.datasets import TestDataset
from torch.utils.data import DataLoader

from rzd_segmentation.utils.visualizer import SegmVisualizer


internal2class_mapping = {0: 0, 1: 6,  2: 7, 3: 10}


def _save_mask(mask, file_path):
    mask_to_save = mask[1:, :]
    mask_to_save = (mask_to_save * 255).astype(np.uint8)
    mask_to_save = mask_to_save.reshape(mask_to_save.shape[1:] + (3, ))
    cv2.imwrite(file_path, mask_to_save)
    
def _load_mask(file_path):
    mask_to_load = cv2.imread(file_path)
    mask_to_load = mask_to_load.reshape((3, ) + mask_to_load.shape[:-1])
    
    mask_to_load = mask_to_load / 255.0
    mask = np.empty((4,) + mask_to_load.shape[1:])
    mask[1:, :] = mask_to_load
    mask[0, :] = 1.0 - (mask[1, :] + mask[2, :] + mask[3, :])
    
    return mask

def proba2pred(mask, shape, resize_first):
    if resize_first:
        mask = skimage.transform.resize(mask, (4, shape[1], shape[0]))
        mask = mask.argmax(axis=0)
    else:
        mask = mask.argmax(axis=0)
        mask = cv2.resize(mask, (shape[0], shape[1]), interpolation=cv2.INTER_NEAREST)

    return mask

def get_tta_model(model):
    tta_transforms = tta.Compose([tta.HorizontalFlip(),])
    model = tta.SegmentationTTAWrapper(model, transforms=tta_transforms, merge_mode='mean')
    return model


def get_model(config, use_tta=False, use_folds=False, fold=None):
    print("tta", use_tta, "folds", use_folds, "fold", fold)
    if use_folds:
        folds = list(range(5))
    elif fold is not None:
        folds = [fold]
    else:
        folds = [0]

    checkpoint_path_patterns = [os.path.join(os.path.join(config.checkpoint_path, 
            f"{config.version_name}_fold{fold}" +  f"_{config.metric_to_monitor}=*")) for fold in folds]

    checkpoints_paths = []
    for pattern in checkpoint_path_patterns:
        checkpoints_paths.append(glob.glob(pattern)[0])
    
    print("loaded checkpoints", checkpoints_paths)

    models = []
    for i, checkpoint_path in enumerate(checkpoints_paths):
        model = create_model(config, num_classes=4)
        model = load_model_weights(model, checkpoint_path)
        model.eval()
        model.cuda()

        if use_tta:
            model = get_tta_model(model)
        models.append(model)
    
    return models


def inference_model(models, X):
    softmax = nn.Softmax(dim=1)

    image_preds = []

    with torch.inference_mode():
        for i, model in enumerate(models):
            image_preds.append(softmax(model(X.cuda())).cpu().numpy().squeeze())

    image_preds = np.mean(np.array(image_preds), axis=0)
    
    return image_preds


def inference_pl(args):
    weights = args.weights

    version_name = ""
    models = []
    datasets = []

    for i, (config_path, model_weight) in enumerate(zip(args.configs, args.weights)):
        config = load_config_file(config_path)
        version_name = version_name + config.version_name + "_"

        models.append(get_model(config, use_tta=args.tta, use_folds=args.folds_ensemble, fold=args.fold))
        datasets.append(TestDataset(args.test_dir, config.val_data.augs_config_path))
    
    dataloaders = [DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8) for dataset in datasets]
    dataloader_iterators = [iter(dataloader) for dataloader in dataloaders[1:]]
    

    output_dir = os.path.join(args.output_dir, 
        version_name + f"folds{args.folds_ensemble}" + f"_tta{args.tta}")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    

    for i, input_0 in tqdm(enumerate(dataloaders[0])):
        shape = input_0[1]
        shape = int(shape[1].item()), int(shape[0].item())
        image_path = input_0[2][0]

        model_preds_0 = inference_model(models[0], input_0[0])
        image_pred = weights[0] * model_preds_0

        for i, dataloader_iterator in enumerate(dataloader_iterators, start=1):
            input_i = next(dataloader_iterator)
            model_preds_i = inference_model(models[i], input_i[0])
            image_pred += weights[i] * model_preds_i


        mask = proba2pred(image_pred, shape, args.resize_first)
        
        mask_remaped = mask.copy()
        for old, new in internal2class_mapping.items():
            mask_remaped[mask == old] = new
        
        cv2.imwrite(os.path.join(output_dir, image_path.split("/")[-1]), mask_remaped)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+')
    parser.add_argument('--weights', nargs='*', type=float, default=[1.0])
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--folds_ensemble', action='store_true')
    parser.add_argument('--resize_first', action='store_true')
    parser.add_argument('--fold', type=int, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    inference_pl(args)
