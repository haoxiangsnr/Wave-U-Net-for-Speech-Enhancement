import torch

def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()

def bce_loss():
    return torch.nn.BCEWithLogitsLoss() # output 0~1