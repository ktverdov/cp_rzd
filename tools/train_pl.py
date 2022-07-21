import os
import argparse
import pytorch_lightning as pl


from rzd_segmentation.pl_training.pl_modules import SegmModule, SegmDataModule
from rzd_segmentation.pl_training.models import load_model_weights
from rzd_segmentation.pl_training.utils import load_config_file


def train_pl(config, args):
    pl.seed_everything(42, workers=True)
    
    if args.checkpoint_name != "":
        config.model.from_checkpoint = args.checkpoint_name
        
    datamodule = SegmDataModule(config, args.fold)
    num_classes = datamodule.get_num_classes()

    model = SegmModule(config, num_classes)

    if config.model.from_checkpoint:
        model = load_model_weights(model, os.path.join(config.checkpoint_path, 
                                                        config.model.from_checkpoint))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(config.checkpoint_path, config.version_name),
        filename='{epoch}'
    )
    
    # checkpoint_callback_loss = pl.callbacks.ModelCheckpoint(dirpath=config.checkpoint_path, 
    #                                          save_top_k=1, 
    #                                          monitor="val_loss", 
    #                                          mode="min",
    #                                          filename=f"{config.version_name}" + "_{val_loss:.2f}")

    checkpoint_callback_metric = pl.callbacks.ModelCheckpoint(dirpath=config.checkpoint_path, 
                                             save_top_k=1, 
                                             monitor=config.metric_to_monitor, 
                                             mode="max",
                                             filename=f"{config.version_name}_fold{args.fold}" + "_{" + f"{config.metric_to_monitor}" + ":.3f}")
    
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", 
                                                     min_delta=0.001, patience=10, 
                                                     verbose=False, 
                                                     mode="min")
    
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name="", version=f"{config.version_name}_fold{args.fold}")

    callbacks_list = [
                        # checkpoint_callback, 
                        # checkpoint_callback_loss, 
                        # early_stop_callback, 
                        lr_monitor_callback, 
                        checkpoint_callback_metric]

    if "use_swa" in config and config.use_swa:
        swa_callback = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=10, device="cpu")
        callbacks_list.append(swa_callback)

    precision = config.precision if "precision" in config else 16
    trainer = pl.Trainer(accelerator='gpu', 
                        devices=1,
                        precision=precision,
                        max_epochs=config.max_epochs,
                        accumulate_grad_batches=config.accumulate_grad_batches, 
                        callbacks=callbacks_list, 
                        logger=logger)
    
    trainer.fit(model, datamodule=datamodule)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration filename', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--checkpoint_name', help='weights to resume', 
                        default="", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config_file(args.config)
    
    train_pl(config, args)
