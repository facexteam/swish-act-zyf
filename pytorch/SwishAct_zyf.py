import torch
import torch.nn as nn
from torch.autograd import Function


class SwishAct(Function):
    @staticmethod
    def forward(ctx, i, beta=1.0):
        # result = i*i.sigmoid()
        result = i*torch.sigmoid(i*beta)
        ctx.save_for_backward(result, i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result+sigmoid_x*(1-result))


swish = SwishAct.apply


class Swish_module(nn.Module):
    def __init__(self, beta=1.0):
        super(Linear, self).__init__()
        self.beta = beta

    def forward(self, x):
        return swish(x, self.beta)


swish_layer = Swish_module()
