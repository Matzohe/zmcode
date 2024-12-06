import torch
from src.Recognize.brain2clip.brain2clip import LinearBrain2CLIP
from src.utils.utils import INIconfig

def test_linear_brain2clip():
    target_layer = ["visual.transformer.resblocks.{}".format(i) for i in range(12)]
    mpl_layer = ["visual.transformers.resblocks.{}.mlp.c_proj".format(i) for i in range(12)]
    config = INIconfig("config/brain2clip_config.cfg")
    brain2clip = LinearBrain2CLIP(config)
    for i in range(1, 9):
        brain2clip.target_layer_linear_fitting(subj=i, voxel_activation_roi="ventral_visual_pathway_roi", target_layer=target_layer)
