import torch
import torch.nn as nn
import os
from dataclasses import dataclass


# ==================================================
# Checkpoint Related Functions
# ==================================================

@dataclass
class CheckPoint:
    # An example of Basic CheckPoint dataclass
    model_state_dict: nn.Module.state_dict
    optimizer_state_dict: torch.optim.Optimizer.state_dict
    batch_idx: int  # The index of the current batch
    epoch_idx: int  # The index of the current epoch


def save_checkpoint(
    config: dataclass,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_idx: int,
    epoch_idx: int,
):
    
    checkpoint = CheckPoint(model.state_dict(), optimizer.state_dict(), batch_idx, epoch_idx)
    checkpoint_save_name = os.path.join(config.TRAINING["checkpoint_dir"].format(type(model).__name__), 
                                        config.TRAINING["checkpoint_save_name"].format(epoch_idx, batch_idx))
    if not os.path.exists(config.TRAINING["checkpoint_dir"].format(type(model).__name__)):
        try:
            os.mkdir(config.TRAINING["checkpoint_dir"].format(type(model).__name__))
        except:
            raise ValueError("Can't create checkpoint dir, you need to create the root path before create the checkpoint file")
    torch.save(checkpoint, checkpoint_save_name)
    print("Checkpoint saved at {}".format(checkpoint_save_name))