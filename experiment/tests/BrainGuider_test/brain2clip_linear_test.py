import torch

if __name__ == "__main__":
    target_layer = ["visual.transformer.resblocks.{}".format(i) for i in range(12)]
    mpl_layer = ["visual.transformers.resblocks.{}.mlp.c_proj".format(i) for i in range(12)]