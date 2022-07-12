import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from .models import create_model
from .optimizers import create_optimizer
from .schedulers import create_scheduler
from .criterions import create_criterion
from .metrics import create_metrics
from .datasets import create_datasets

from rzd_segmentation.tile_inference import inference_on_image


class SegmModule(pl.LightningModule):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        self.model = create_model(self.config, num_classes)
        self.criterion = create_criterion(self.config, num_classes)
        self.metrics = create_metrics(self.config)

        self.tile_inference = True if "tile_inference" in self.config and self.config.tile_inference else False
        print(self.tile_inference)
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y.long())

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.tile_inference:
            logits = torch.tensor(inference_on_image(self.model, x.cpu().numpy().squeeze(), self.num_classes)).cuda()
            logits = torch.moveaxis(logits, 2, 0)
            logits = logits.unsqueeze(0)
        else:
            logits = self.model(x)

        loss = self.criterion(logits, y.long())

        for metric in self.metrics:
            metric.update(logits=logits.cpu().numpy(), 
                            targets=y.cpu().numpy())
    
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
    def validation_epoch_end(self, validation_step_outputs):
        for metric in self.metrics:
            metric_result = metric.compute()
            metric.reset()
            self.log(f"{metric.name}", metric_result, prog_bar=True)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.config, self.model)
        scheduler = create_scheduler(self.config, optimizer)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}]
        
        
class SegmDataModule(pl.LightningDataModule):
    def __init__(self, config, fold):
        super().__init__()
        self.config = config
        self.datasets = create_datasets(config, fold)

    def get_num_classes(self):
        return self.datasets["train"].get_num_classes()
        
    def train_dataloader(self):
        return DataLoader(self.datasets["train"], 
                          batch_size=self.config.train_data.batch_size, 
                          shuffle=True, 
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], 
                          batch_size=self.config.val_data.batch_size, 
                          shuffle=False, 
                          num_workers=8)
