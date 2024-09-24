import torch
import torch.nn as nn
from typing import Union
from torch.utils.data import DataLoader
import torchextractor as tx
from collections import OrderedDict
from tqdm import tqdm
import os
from ..CKA import CFA


class cfaAnalysis(nn.Module):

    # TODO: Sove the memory assumption problem, and complex the cfa analysis function

    def __init__(self, model_A, model_B, config):
        # At least one model
        super().__init__()
        self.config = config
        self.device = config.INFERENCE['device']
        self.layers_A = eval(config.CKA["model_a_layers"])
        self.layers_B = eval(config.CKA["model_b_layers"])
        self.model_A = model_A
        self.model_B = model_B
        self.middle_info_save_root = config.CKA["middle_activation_save_root"]
        self.middle_info_save_name = config.CKA["middle_activation_save_name"]
        self.kernel = config.CKA['kernel']

    def infer(self, input):
        
        tx1 = tx.Extractor(self.model_A, self.layers_A)
        tx2 = tx.Extractor(self.model_B, self.layers_B)

        features_A = tx1(input)
        features_B = tx2(input)
        data1 = features_A[1]
        data2 = features_B[1]
        torch.cuda.empty_cache()
        del tx1
        del tx2
        del features_A
        del features_B
        return data1, data2
    
    def cfaCulculation(self, matrix_A: torch.tensor, matrix_B: torch.tensor):
        output_info = CFA(matrix_A, matrix_B, kernel=self.kernel)
        return output_info.detach().cpu()
    
    def forward(self, dataloader):

        for batch_number, (data, _) in enumerate(tqdm(dataloader)):
            data = data.to(device=self.device)
            features_A, features_B = self.infer(data)
            
            for layer in self.layers_A:
                save_root = self.middle_info_save_root.format(type(self.model_A).__name__ + "A", layer)
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                save_name = self.middle_info_save_name.format(batch_number)
                save_file = os.path.join(save_root, save_name)
                torch.save(features_A[layer], save_file)
            
            for layer in self.layers_B:
                save_root = self.middle_info_save_root.format(type(self.model_B).__name__ + "B", layer)
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                save_name = self.middle_info_save_name.format(batch_number)
                save_file = os.path.join(save_root, save_name)
                torch.save(features_B[layer], save_file)
            del features_A
            del features_B
            del data

        # for key, value in dict_A.items():
        #     middle_data = torch.cat(value, dim=0)
        #     dict_A[key] = middle_data.view(middle_data.shape[0], -1)
        
        # for key, value in dict_B.items():
        #     middle_data = torch.cat(value, dim=0)
        #     dict_B[key] = middle_data.view(middle_data.shape[0], -1)

        # output_cfa = []

        # for _, value_A in tqdm(dict_A.items(), desc="CFA calculating"):
        #     middle_similar = []
        #     for _, value_B in dict_B.items():
        #         middle_similar.append(self.cfaCulculation(value_A, value_B))
        #     output_cfa.append(middle_similar)

        # return output_cfa
