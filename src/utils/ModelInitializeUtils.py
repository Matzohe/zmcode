# ==================================================
# Functions for Model Parameter Initialization
# ==================================================

import torch
import torch.nn as nn
import torch.nn.init as init


# When use the weight_initialize function
# we can use model.apply(lambda module: weight_initialize(module, mean=0.0, std=0.02))
# the std is equal with the sqrt of input dim d (768-1600)

def LLM_weight_initialize(model, config):
    if isinstance(model, nn.Linear):
        std = model.in_features ** -0.5
        if hasattr(model, 'WITH_RESIDUAL'):
            std *= (2 * config["MODEL"]["layers"]) ** 0.5
        torch.nn.init.normal_(model.weight, mean=float(config["TRAINING"]["init_mean"]), std=std)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)
    elif isinstance(model, nn.Embedding):
        # we can initialize it with std=0.02
        # std = model.embedding_dim ** -0.5
        # Initialize is different with different embeddings

        if hasattr(model, 'TOKEN_EMBEDDING'):
            torch.nn.init.normal_(model.weight, mean=float(config["TRAINING"]["init_mean"]), std=0.02)
        elif hasattr(model, 'POSITION_EMBEDDING'):
            torch.nn.init.normal_(model.weight, mean=float(config["TRAINING"]["init_mean"]), std=0.01)

def CNN_initialize(model):
    # The Only thing we need to do is set the output barch normalization's gama to zero
    # As torch initialize the CNN model with the Kaiming initialization, we don't need to initialize it
    if isinstance(model, nn.BatchNorm2d) and hasattr(model, "WITH_RESIDUAL"):
        torch.nn.init.zeros_(model.weight)