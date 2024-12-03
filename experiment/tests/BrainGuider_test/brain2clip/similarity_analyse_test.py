from src.utils.utils import INIconfig
from src.Recognize.clip2brain.VisualPathAnalysis import VisualPathAnalysis

def test_visual_path_analysis():
    target_layers = ["visual.transformer.resblocks.{}".format(i) for i in range(12)]
    config = INIconfig("config/brain2clip_config.cfg")
    vpa = VisualPathAnalysis(config)
    for target_layer in target_layers:
        vpa.extract_similarity(target_layer=target_layer)