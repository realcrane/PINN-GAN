import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTKLinear(nn.Linear):
    """Linear layer with the explicit NTK parameterization W / sqrt(fan_in).

    PyTorch's regular ``Linear`` already initializes weights at a scale that
    depends on ``fan_in``.  For an explicit NTK parameterization we instead
    keep an O(1) raw weight parameter and apply the 1/sqrt(fan_in) factor in
    the forward pass.
    """

    def __init__(self, in_features, out_features, bias=True, bias_std=0.0):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            if bias_std > 0:
                nn.init.normal_(self.bias, mean=0.0, std=bias_std)
            else:
                nn.init.zeros_(self.bias)
        self.ntk_scale = 1.0 / math.sqrt(in_features)

    def forward(self, x):
        return F.linear(x, self.weight * self.ntk_scale, self.bias)

class ResidualBlock(nn.Module):
    """ Most basic residual block
        https://arxiv.org/pdf/1512.03385.pdf : Equation #1
    """

    def __init__(
        self,
        n_units,
        activation,
        spectral_norm=False,
        ntk_parameterization=False,
        ntk_bias_std=0.0,
    ):
        super(ResidualBlock, self).__init__()

        norm = lambda x: nn.utils.spectral_norm(x) if spectral_norm else x
        linear = NTKLinear if ntk_parameterization else nn.Linear
        linear_kwargs = (
            {'bias_std': ntk_bias_std} if ntk_parameterization else {}
        )

        self.activation = activation
        self.l1 = norm(linear(n_units, n_units, **linear_kwargs))
        self.l2 = norm(linear(n_units, n_units, **linear_kwargs))

    def forward(self, x):
        return self.activation(
            self.l2(self.activation(self.l1(x))) + x
        )

class MLP(nn.Module):
    """
    Multi-layer perceptron

    Optionally use residual connections
    Default is sigmoid output
    """
    def __init__(self, in_dim=1, out_dim=1, n_hidden_units=20, n_hidden_layers=2,
        activation=nn.Tanh(), residual=False, regress=False, spectral_norm=False, 
        n_heads=1, pretrained=False, ntk_parameterization=False,
        ntk_bias_std=0.0):

        super().__init__()

        if isinstance(activation, str):
            activation = eval('nn.'+activation+'()')

        norm = lambda x: nn.utils.spectral_norm(x) if spectral_norm else x
        linear = NTKLinear if ntk_parameterization else nn.Linear
        linear_kwargs = (
            {'bias_std': ntk_bias_std} if ntk_parameterization else {}
        )

        # input
        self.layers = nn.ModuleList()
        self.layers.append(norm(linear(
            in_dim*n_heads, n_hidden_units, **linear_kwargs
        )))
        self.layers.append(activation)

        # hidden
        for l in range(n_hidden_layers):
            if residual:
                self.layers.append(ResidualBlock(
                    n_hidden_units,
                    activation,
                    spectral_norm=spectral_norm,
                    ntk_parameterization=ntk_parameterization,
                    ntk_bias_std=ntk_bias_std,
                ))
            else:
                self.layers.append(norm(linear(
                    n_hidden_units, n_hidden_units, **linear_kwargs
                )))
                self.layers.append(activation)

        # output
        self.layers.append(norm(linear(
            n_hidden_units, out_dim, **linear_kwargs
        )))
        if not regress:
            # For WGAN: Discriminator should be unbounded (i.e. no sigmoid / regress = True)
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class MultiHeadGen(nn.Module):
    """
    Multi-layer perceptron

    Optionally use residual connections
    Default is sigmoid output
    """
    def __init__(self, in_dim=1, out_dim=1, n_hidden_units=20, n_hidden_layers=2,
        activation=nn.Tanh(), residual=False, regress=False, spectral_norm=False,
        n_heads=1, n_head_units=20, pretrained=False,
        ntk_parameterization=False, ntk_bias_std=0.0):

        super().__init__()
        self.n_heads = n_heads
        
        if isinstance(activation, str):
            activation = eval('nn.'+activation+'()')

        linear = NTKLinear if ntk_parameterization else nn.Linear
        linear_kwargs = (
            {'bias_std': ntk_bias_std} if ntk_parameterization else {}
        )

        # input
        self.layers = nn.ModuleList()
        self.layers.append(linear(
            in_dim, n_hidden_units, **linear_kwargs
        ))
        self.layers.append(activation)

        # hidden
        for l in range(n_hidden_layers):
            if residual:
                self.layers.append(ResidualBlock(
                    n_hidden_units,
                    activation,
                    spectral_norm=spectral_norm,
                    ntk_parameterization=ntk_parameterization,
                    ntk_bias_std=ntk_bias_std,
                ))
            else:
                self.layers.append(linear(
                    n_hidden_units, n_hidden_units, **linear_kwargs
                ))
                self.layers.append(activation)

        # heads
        # self.heads = nn.ModuleList([nn.ModuleList([norm(nn.Linear(n_hidden_units, n_hidden_units))])]) 
        # self.heads.extend([nn.ModuleList([norm(nn.Linear(n_hidden_units, n_hidden_units))]) for l in range(self.n_heads-1)])
        # for head in self.heads:
        #     head.append(activation)
        #     head.append(norm(nn.Linear(n_hidden_units, n_hidden_units)))
        self.heads = nn.ModuleList([linear(
            n_hidden_units, n_head_units, **linear_kwargs
        )])
        self.heads.extend([
            linear(n_hidden_units, n_head_units, **linear_kwargs)
            for l in range(self.n_heads-1)
        ])

        # output
        self.outputs = nn.ModuleList([linear(
            n_head_units, out_dim, **linear_kwargs
        )])
        self.outputs.extend([
            linear(n_head_units, out_dim, **linear_kwargs)
            for l in range(self.n_heads-1)
        ])

    def forward(self, x):
        if self.n_heads > 1:
            d = {}
            for i in range(len(self.layers)):
                x = self.layers[i](x)
            for j in range(self.n_heads):
                x1 = self.heads[j](x)
                # for k in range(len(self.heads[j])):
                #     x1 = self.heads[j][k](x)
                x1 = self.outputs[j](x1)
                d[j] = x1
            return d
        else:
            for i in range(len(self.layers)):
                x = self.layers[i](x)
            x1 = self.heads[0](x)
            x1 = self.outputs[0](x1)
            return x1

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, input):
        return input * torch.sigmoid(self.beta * input)

class TorchSin(nn.Module):
    """
    Sin activation function
    """
    def __init__(self):
        super(TorchSin, self).__init__()

    def forward(self, x):
        return torch.sin(x)
