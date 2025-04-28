import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Kernel same as in SpatialPCA
'''
class Kernel(nn.Module):
    def __init__(self, kernel_type='Gaussian', scale=1., fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(Kernel, self).__init__()
        self.kernel_type = kernel_type
        self.fixed_scale = fixed_scale
        if self.fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        d = torch.cdist(x, y, p=2) # Euclidean distance
        # kernel type
        if self.kernel_type == 'Gaussian':
            if self.fixed_scale:
                res = torch.exp(-1*torch.square(d)/self.scale)
            else:
                res = torch.exp(-1*torch.square(d)/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        if self.kernel_type == 'Cauchy':
            if self.fixed_scale:
                res = 1 / (1 + 1*torch.square(d)/self.scale)
            else:
                res = 1 / (1 + 1*torch.square(d)/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        if self.kernel_type == 'Quadratic':
            if self.fixed_scale:
                res = 1 - 1 * torch.square(d) / (1 * torch.square(d) + self.scale)
            else:
                res = 1 - 1 * torch.square(d) / (1 * torch.square(d) + torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        if self.kernel_type == 'Matern':
            
            pass
        return res

    def print_scale(self):
        print(self.scale)