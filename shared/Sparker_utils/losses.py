import torch

# losses                                                                                                 
def NPLMLoss(true, pred):
    f   = pred[:, 0]
    y   = true[:, 0]
    w   = true[:, 1]
    return torch.sum((1-y)*w*(torch.exp(f)-1) - y*w*(f))

def MSELoss(true, pred):
    f   = 1./(1+torch.exp(-1*pred[:, 0])) # sigmoid  
    y   = true[:, 0]
    w   = true[:, 1]
    return torch.sum(((f-y)**2)*w)

def BCELoss(true, pred):
    f   = 1./(1+torch.exp(-1*pred[:, 0])) # sigmoid 
    y   = true[:, 0]
    w   = true[:, 1]
    return torch.sum(-1*w*((1-y)*torch.log(1-f)+y*torch.log(f)))
