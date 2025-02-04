import torch
import torch.linalg
import torch.nn as nn

def HiFAR_penalty_term(model):
    penalty = 0
    device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            out_channels, in_channels, h, w = module.weight.size()
            weight_fft = torch.fft.fft2(module.weight, dim=(-2, -1),norm='forward')
            u = torch.arange(h).to(device)
            u = u.type(torch.float)
            v = torch.arange(w).to(device)
            v = v.type(torch.float)
            v = torch.reshape(v,(w,1))
            freq_weight = u.pow(2) + v.pow(2)
            freq_weight_norm = freq_weight/torch.linalg.matrix_norm(freq_weight)
            penalty += ((freq_weight_norm * weight_fft).abs().pow(2)).sum()
            break
    return penalty
