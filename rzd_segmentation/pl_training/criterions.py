import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, LovaszLoss, TverskyLoss
from torch.nn.modules.loss import _Loss


def get_loss(loss_type, n_classes, **params):
    if loss_type == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(**params)
    elif loss_type == "DiceLoss":
        criterion = DiceLoss(**params)
    elif loss_type == "FocalLoss":
        criterion = FocalLoss(**params)
    elif loss_type == "LovaszLoss":
        criterion = LovaszLoss(**params)
    elif loss_type == "TverskyLoss":
        criterion = TverskyLoss(**params)
    else:
        raise ValueError("Error in defining loss")
        
    return criterion

def create_criterion(config, n_classes):
    # return get_loss(config.loss[0].type, n_classes)
    losses = []
    weights = []

    for loss_item in config.loss:
        parameters = loss_item.parameters if "parameters" in loss_item else {}
        losses.append(get_loss(loss_item.type, n_classes, **parameters))
        weights.append(loss_item.weight)
    
    loss = JointLoss(losses, weights)
    
    return loss


class JointLoss(_Loss):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, *input):
        result = self.losses[0](*input) * self.weights[0]
        
        for loss, weight in zip(self.losses[1:], self.weights[1:]):
            result += loss(*input) * weight

        return result
