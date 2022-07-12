import yaml
from addict import Dict
import importlib
import albumentations as A
from typing import Any

def load_yaml(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_config_file(config_path):
    config = load_yaml(config_path)
    config = Dict(config)
    return config


def load_obj(obj_path: str, default_obj_path: str = '') -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def load_augs(cfg) -> A.Compose:
    """
    Load albumentations
    Args:
        cfg:
    Returns:
        compose object
    """
    augs = []
    
    for a in cfg:
        if a['class_name'] == 'albumentations.OneOf':
            small_augs = []
            
            for small_aug in a['augs']:
                aug = load_obj(small_aug['class_name'])(**small_aug['params'])
                small_augs.append(aug)
            
            aug = load_obj(a['class_name'])(small_augs, **a["params"])
            augs.append(aug)
        else:
            aug = load_obj(a['class_name'])(**a['params'])
            augs.append(aug)

    return A.Compose(augs)
