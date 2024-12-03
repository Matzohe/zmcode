import torch


class VisualPathAnalysis:
    def __init__(self, config):
        self.config = config
        
        
        self.linear_weight = None
        self.target_layer = None
        