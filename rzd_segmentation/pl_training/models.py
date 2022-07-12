import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Parameter
# import torchvision
# import math
# import timm

import segmentation_models_pytorch as smp


def create_model(config, num_classes):
    if config.model.type == "smp":
        model = create_smp_model(config, num_classes)

    return model

def create_smp_model(config, num_classes):
    if config.model.arch == "unet":
        model = smp.Unet(**config.model.parameters, classes=num_classes)
    elif config.model.arch == "deeplabv3":
        model = smp.DeepLabV3(**config.model.parameters, classes=num_classes)
    elif config.model.arch == "fpn":
        model = smp.FPN(**config.model.parameters, classes=num_classes)
    elif config.model.arch == "pan":
        model = smp.PAN(**config.model.parameters, classes=num_classes)
    elif config.model.arch == "linknet":
        model = smp.Linknet(**config.model.parameters, classes=num_classes)
    elif config.model.arch == "pspnet":
        model = smp.PSPNet(**config.model.parameters, classes=num_classes)

    return model

def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = {k.partition('model.')[2]: v for k,v in checkpoint['state_dict'].items()}
    # state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    print(f"Loaded from checkpoint {checkpoint_path}")

    return model

