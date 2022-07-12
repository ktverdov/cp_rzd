import numpy as np
from collections import Counter
import sklearn.metrics as sklearn_metrics
import torch
import segmentation_models_pytorch as smp


class BaseMetric:
    def __init__(self, **kwargs):
        self._name = kwargs["name"]
    
    @property
    def name(self):
        return self._name

    def update(self, **kwargs):
        pass

    def compute(self):
        pass
    
    def reset(self):
        pass


# def iou(pred, mask):
#     intersection = np.logical_and(pred, mask)
#     union = np.logical_or(pred, mask)
#     iou_score = np.sum(intersection) / np.sum(union)
#     return iou_score


class IOUMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iou_scores = []
        
    def update(self, **kwargs):
        if "logits" in kwargs:
            preds = kwargs["logits"].argmax(axis=1)
        else:
            preds = kwargs["pred_class"]

        for pred, mask in zip(preds, kwargs["targets"]):
            tp, fp, fn, tn = smp.metrics.get_stats(torch.tensor(pred).long(), 
                                                    torch.tensor(mask).long(), 
                                                    mode='multiclass', 
                                                    num_classes=4)

            # iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro",
                                zero_division=1.0)

            self.iou_scores.append(iou_score)
        
    def compute(self):
        return np.mean(self.iou_scores)

    def reset(self):
        self.iou_scores = []


def create_metrics(config):
    metrics = []

    for metric_name in config.metrics:
        metrics.append(create_metric(metric_name))
    
    return metrics

def create_metric(name):
    if name == "iou": 
        return IOUMetric(name=name)
