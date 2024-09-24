import torch
import torch.nn as nn
from typing import Union
from torch.utils.data import DataLoader
import torchextractor as tx
from collections import OrderedDict

from ..CFA import CFA


class cfaAnalysis(nn.Module):
    def __init__(self, model_A, model_B, model_A_layers, model_B_layers, kernel="linear"):
        # At least one model
        super().__init__()
        self.model_A = tx.Extractor(model_A.eval(), layers=model_A_layers)
        self.model_B = tx.Extractor(model_B.eval(), layers=model_B_layers)
        self.layers_A = model_A_layers
        self.layers_B = model_B_layers
        self.kernel = kernel

    def infer(self, input):
        _, features_A = self.model_A(input)
        _, features_B = self.model_B(input)
        return features_A, features_B
    
    def cfaCulculation(self, matrix_A: torch.tensor, matrix_B: torch.tensor):
        return CFA(matrix_A, matrix_B, kernel=self.kernel)
    
    def forward(self, dataloader):

        dict_A = OrderedDict({
            layer:[] for layer in self.layers_A
        })

        dict_B = OrderedDict({
            layer:[] for layer in self.layers_B
        })

        for data, _ in dataloader:
            features_A, features_B = self.infer(data)
            for layer in self.layers_A:
                dict_A[layer].append(features_A[layer])
            for layer in self.layers_B:
                dict_B[layer].append(features_B[layer])

        for key, value in dict_A.items():
            middle_data = torch.cat(value, dim=0)
            dict_A[key] = middle_data.view(middle_data.shape[0], -1)
        
        for key, value in dict_B.items():
            middle_data = torch.cat(value, dim=0)
            dict_B[key] = middle_data.view(middle_data.shape[0], -1)

        output_cfa = []

        for _, value_A in dict_A.items():
            middle_similar = []
            for _, value_B in dict_B.items():
                middle_similar.append(self.cfaCulculation(value_A, value_B))
            output_cfa.append(middle_similar)

        return output_cfa
