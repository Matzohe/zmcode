from src.utils.utils import INIconfig
from src.Recognize.brain2clip.VisualPathAnalysis import VisualPathAnalysis
import matplotlib.pyplot as plt


def visual_path_test():
    target_layers = ["visual.transformer.resblocks.{}".format(i) for i in range(12)]
    config = INIconfig("config/brain2clip_config.cfg")
    vpa = VisualPathAnalysis(config)
    output = vpa.multi_process_extract_target_voxels(subj=1, target_layer=target_layers[0])
    plt.plot([i for i in range(len(output.cpu()))], output.cpu())
    plt.show()