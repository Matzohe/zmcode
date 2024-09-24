import torch
from typing import Union
from torch_cka import CKA as _CKA


# ==================================================
# CFA Function use for PostHoc explainable
# ==================================================

def CKA(
        matrix_A: torch.tensor,
        matrix_B: torch.tensor,
        kernel: str,
    ):
    if kernel == "linear":
        middle_A = torch.matmul(matrix_A, matrix_A.T)
        middle_B = torch.matmul(matrix_B, matrix_B.T)
    elif kernel == "RBF":
        # TODO: Fix the RBF kernel
        raise ValueError("RBF kernel function is not supported yet, please use linear kernel instead")
    else:
        raise ValueError(f"kernel {kernel} is not supported yet")
    
    return HSIC(middle_A, middle_B) / torch.sqrt(HSIC(matrix_A, matrix_A) * HSIC(matrix_B, matrix_B))


def HSIC(
        middle_A: torch.tensor, 
        middle_B: torch.tensor,
    ):

    H = torch.eye(middle_A) - torch.tensor(1 / middle_A.shape[-1], dtype=middle_A.dtype, device=middle_A.device)
    w = torch.tensor(1 / (middle_A.shape[-1] ** 2 - 1), dtype=middle_A.dtype, device=middle_A.device)
    x = torch.matmul(middle_A, H)
    x = torch.matmul(x, middle_B)
    x = torch.matmul(x, H)

    return torch.trace(x) * w


def CKA_Function(model_A, model_B, _data_loader, model_A_name: Union[str] = None, model_B_name: Union[str] = None, device='cpu'):
    if model_A_name is None:
        model_A_name = type(model_A).__name__ + "A"
    if model_B_name is None:
        model_B_name = type(model_B).__name__ + "B"

    cka = _CKA(model_A, model_B, model1_name=model_A_name, model2_name=model_B_name, device=device)

    cka.compare(_data_loader)
    result = cka.export()

    return result