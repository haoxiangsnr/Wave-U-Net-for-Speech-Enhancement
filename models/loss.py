import torch

def mse_loss(output, target):
    loss =  torch.nn.MSELoss()
    return loss(output, target)

def l1_loss(output, target):
    loss = torch.nn.L1Loss()
    return loss(output, target)

def bce_loss(output, target):
    loss = torch.nn.BCELoss()
    return loss(output, target)