import torch

def create_scheduler(config, optimizer, total_steps):
    if "scheduler" in config:
        if config.scheduler.type == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **config.scheduler.parameters, 
                total_steps=total_steps)
        else:
            raise ValueError("Error in defining scheduler")

        return scheduler
    
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, 
        threshold=0.01, threshold_mode='abs', cooldown=1, min_lr=3e-6, eps=1e-8, verbose=True)
