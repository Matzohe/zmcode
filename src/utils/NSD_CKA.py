import torch
from typing import Union

class CKAforNSD:
    def __init__(self, config):
        self.config = config
        self.nsd_response_data_root = config.NSD['pure_response_save_root']
        self.cka_batch_size = int(config.NSD['cka_batch_size'])

    def process(self, subj1, roi1, subj2: Union[str] = None, roi2: Union[str] = None):
        
        # load data
        response1 = torch.load(self.nsd_response_data_root.format(subj1, roi1))
        if subj2 is None or roi2 is None:
            response2 = torch.load(self.nsd_response_data_root.format(subj1, roi1))
        else:
            response2 = torch.load(self.nsd_response_data_root.format(subj2, roi2))
        
        # compute CKA, to compute it by batch
        batch_num = response1.shape[0] // self.cka_batch_size
        if batch_num * self.cka_batch_size < response1.shape[0]:
            batch_num += 1
        cka_matrix = torch.zeros(size=[response1.shape[1], response2.shape[1], 3])

        for batch in range(batch_num):
            if batch == batch_num - 1:
                size = response1.shape[0] - batch * self.cka_batch_size
            else:
                size = self.cka_batch_size
            for i in range(response1.shape[1]):

                data1 = response1[self.cka_batch_size * batch: self.cka_batch_size * batch + size, i, :]
                data1.view(size, -1)
                K = data1 @ data1.t()
                K.fill_diagonal_(0.0)
                cka_matrix[i, :, 0] += self._HSIC(K, K) / size

                for j in range(response2.shape[1]):

                    data2 = response2[self.cka_batch_size * batch: self.cka_batch_size * batch + size, j, :]
                    data2.view(size, -1)
                    L = data2 @ data2.t()
                    L.fill_diagonal_(0.0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    cka_matrix[i, j, 1] += self._HSIC(K, L)
                    cka_matrix[i, j, 2] += self._HSIC(L, L)
        
        cka_matrix = cka_matrix[:, :, 1] / (cka_matrix[:, :, 0].sqrt() *
                                                        cka_matrix[:, :, 2].sqrt())

        return cka_matrix

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()
