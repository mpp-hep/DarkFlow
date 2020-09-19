import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MaskedLinear(nn.Module):

    def __init__(self, in_features, out_features, diagonal_zeros=False, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask.cuda()
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features
        assert n_in % n_out == 0 or n_out % n_in == 0

        mask = np.ones((n_in, n_out), dtype=np.float32)
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i + 1:, i * k:(i + 1) * k] = 0
                if self.diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[(i + 1) * k:, i:i + 1] = 0
                if self.diagonal_zeros:
                    mask[i * k:(i + 1) * k:, i:i + 1] = 0
        return mask

    def forward(self, x):
        output = x.mm(self.mask * self.weight)

        if self.bias is not None:
            return output.add(self.bias.expand_as(output))
        else:
            return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', diagonal_zeros=' \
            + str(self.diagonal_zeros) + ', bias=' \
            + str(bias) + ')'


class MaskedConv2d(nn.Module):

    def __init__(self, in_features, out_features, size_kernel=(3, 3), diagonal_zeros=False, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.size_kernel = size_kernel
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(out_features, in_features, *self.size_kernel))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask.cuda()
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features

        assert n_out % n_in == 0 or n_in % n_out == 0, "%d - %d" % (n_in, n_out)

        # Build autoregressive mask
        l = (self.size_kernel[0] - 1) // 2
        m = (self.size_kernel[1] - 1) // 2
        mask = np.ones((n_out, n_in, self.size_kernel[0], self.size_kernel[1]), dtype=np.float32)
        mask[:, :, :l, :] = 0
        mask[:, :, l, :m] = 0

        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i * k:(i + 1) * k, i + 1:, l, m] = 0
                if self.diagonal_zeros:
                    mask[i * k:(i + 1) * k, i:i + 1, l, m] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[i:i + 1, (i + 1) * k:, l, m] = 0
                if self.diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k:, l, m] = 0

        return mask

    def forward(self, x):
        output = F.conv2d(x, self.mask * self.weight, bias=self.bias, padding=(1, 1))
        return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', diagonal_zeros=' \
            + str(self.diagonal_zeros) + ', bias=' \
            + str(bias) + ', size_kernel=' \
            + str(self.size_kernel) + ')'


class CNN_Flow_Layer(nn.Module):
    def __init__(self, dim, kernel_size, dilation, test_mode=0, rescale=True, skip=True):
        super(CNN_Flow_Layer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.test_mode = test_mode
        self.rescale = rescale
        self.skip = skip
        self.usecuda = True

        if self.rescale: # last layer of flow needs to account for the scale of target variable
            self.lmbd = nn.Parameter(torch.FloatTensor(self.dim).normal_().cuda())
        
        self.conv1d = nn.Conv1d(1, 1, kernel_size, dilation=dilation)
            
    def forward(self, x):

        # pad zero to the right
        padded_x = F.pad(x, (0, (self.kernel_size-1) * self.dilation))

        conv1d = self.conv1d(padded_x.unsqueeze(1)).squeeze() #(bs, 1, width)

        w = self.conv1d.weight.squeeze()        

        # make sure u[i]w[0] >= -1 to ensure invertibility for h(x)=tanh(x) and with skip

        neg_slope = 1e-2
        activation = F.leaky_relu(conv1d, negative_slope=neg_slope)
        activation_gradient = ((activation>=0).float() + (activation<0).float()*neg_slope)

        # for 0<=h'(x)<=1, ensure u*w[0]>-1
        scale = (w[0] == 0).float() * self.lmbd \
                +(w[0] > 0).float() * (-1./w[0] + F.softplus(self.lmbd)) \
                +(w[0] < 0).float() * (-1./w[0] - F.softplus(self.lmbd))

        
        if self.rescale:
            if self.test_mode:
                activation = activation.unsqueeze(dim=0)
                activation_gradient = activation_gradient.unsqueeze(dim=0)
            output = activation.mm(torch.diag(scale))
            activation_gradient = activation_gradient.mm(torch.diag(scale))
        else:
            output = activation

        if self.skip:
            output = output + x
            logdet = torch.log(torch.abs(activation_gradient*w[0] + 1)).sum(1)
        
        else:
            logdet = torch.log(torch.abs(activation_gradient*w[0])).sum(1)

        return output, logdet
        

class Dilation_Block(nn.Module):
    def __init__(self, dim, kernel_size, test_mode=0):
        super(Dilation_Block, self).__init__()

        self.block = nn.ModuleList()
        i = 0
        while 2**i <= dim:
            conv1d = CNN_Flow_Layer(dim, kernel_size, dilation=2**i, test_mode=test_mode)
            self.block.append(conv1d)
            i+= 1

    def forward(self, x):
        logdetSum = 0
        output = x
        for i in range(len(self.block)):
            output, logdet = self.block[i](output)
            logdetSum += logdet

        return output, logdetSum
        


