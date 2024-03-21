import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import ScriptModule, script_method
from torch import Tensor
from typing import Tuple

class ICNN(ScriptModule):
    def __init__(self, 
                 M: int, 
                 N: int, 
                 kernel_size: Tuple[int, int], 
                 stride: Tuple[int, int], 
                 padding: Tuple[int, int], 
                 out_size: int, 
                 groups: int = 1,
                 norm_template: float = 1, 
                 device: torch.device = None):
        super(ICNN, self).__init__()
        self.add_conv = nn.Conv2d(in_channels=M, out_channels=N, kernel_size=kernel_size, groups=groups, stride=stride, padding=padding)
        self.out_size = out_size

        mus = torch.FloatTensor([[i, j] for i in range(self.out_size) for j in range(self.out_size)])
        templates = torch.zeros(mus.size(0), self.out_size, self.out_size)

        n_square = self.out_size * self.out_size
        tau = 0.5 / n_square
        alpha = n_square / (1 + n_square)
        beta = 4

        for k in range(templates.size(0)):
            for i in range(self.out_size):
                for j in range(self.out_size):
                    if k < templates.size(0) - 1:  # positive templates
                        norm = (torch.FloatTensor([i, j]) - mus[k]).norm(norm_template, -1)
                        out = tau * torch.clamp(1 - beta * norm / self.out_size, min=-1)
                        templates[k, i, j] = float(out)

        self.templates_f = templates.requires_grad_(False).to(device)
        neg_template = -tau * torch.ones(1, self.out_size, self.out_size)
        templates = torch.cat([templates, neg_template], 0)
        self.templates_b = templates.requires_grad_(False).to(device)

        p_T = [alpha / n_square for _ in range(n_square)]
        p_T.append(1 - alpha)
        self.p_T = torch.FloatTensor(p_T).requires_grad_(False).to(device)

    @script_method
    def get_masked_output(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        indices = F.max_pool2d(x, self.out_size, return_indices=True)[1].squeeze()
        selected_templates = torch.stack([self.templates_f[i] for i in indices], dim=0)
        x_masked = F.relu(x * selected_templates)
        return x_masked, selected_templates

    @script_method
    def compute_local_loss(self, x: Tensor) -> Tensor:  
        tr_x_T = torch.einsum('bcwh,twh->cbt', x, self.templates_b)
        p_x_T = F.softmax(tr_x_T, dim=1)

        p_x = (self.p_T[None, None, :] * p_x_T).sum(-1)
        p_x_T_log = (p_x_T * torch.log(p_x_T/p_x[:, :, None])).sum(1)
        loss = - (self.p_T[None, :] * p_x_T_log).sum(-1)
        return loss

    @script_method
    def forward(self, x: Tensor, train: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x1, _ = self.get_masked_output(x) 
        # x1 *= x
        x = self.add_conv(x1)
        x2, _ = self.get_masked_output(x)
        # x2 *= x

        x = x2

        loss_1 = self.compute_local_loss(x1)
        loss_2 = self.compute_local_loss(x2)
        return x, loss_1, loss_2
    