import torch

def r2_score(Real, Pred):
    SSres = torch.mean((Real - Pred) ** 2, dim=0)
    SStot = torch.var(Real, dim=0)
    return torch.nan_to_num(1 - SSres / SStot)