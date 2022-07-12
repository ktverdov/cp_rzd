import torch.optim as optim


def create_optimizer(config, model):
    if config.optimizer.type == "Adam":
        optimizer = optim.Adam(model.parameters(), **config.optimizer.parameters)
    else:
        raise ValueError("Error in defining optimizer")
        
    if config.optimizer.from_checkpoint:
        checkpoint = torch.load(config.optimizer.from_checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return optimizer
