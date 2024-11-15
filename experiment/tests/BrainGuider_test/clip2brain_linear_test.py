from src.Recognize.clip2brain.clip2brain import LinearClip2Brain
from src.utils.utils import INIconfig
from torch.utils.tensorboard import SummaryWriter


def test_linear_clip2brain():
    config = INIconfig("config/brainGuide_config.cfg")
    linear_clip2brain = LinearClip2Brain(config)
    summaryWriter = SummaryWriter()
    linear_clip2brain.linear_fitting(summary_writer=summaryWriter)