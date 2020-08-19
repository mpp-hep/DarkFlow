"""
Collection of flow strategies
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from networks.layers import MaskedConv2d, MaskedLinear


class Planar(nn.Module):

    def __init__(self):

        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """

        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b):

        zk = zk.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return z, log_det_jacobian


class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):

        super(Sylvester, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs

        self.h = nn.Tanh()

        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        r1_hat = r1
        r2_hat = r2

        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)

        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):

        super(TriangularSylvester, self).__init__()

        self.z_size = z_size
        self.h = nn.Tanh()

        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk

        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.h(r2qzb), r1.transpose(2, 1))

        if permute_z is not None:
            # permute order of z again back again
            z = z[:, :, permute_z]

        z += zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class IAF(nn.Module):

    def __init__(self, z_size, num_flows=2, num_hidden=0, h_size=50, forget_bias=1., conv2d=False):
        super(IAF, self).__init__()
        self.z_size = z_size
        self.num_flows = num_flows
        self.num_hidden = num_hidden
        self.h_size = h_size
        self.conv2d = conv2d
        if not conv2d:
            ar_layer = MaskedLinear
        else:
            ar_layer = MaskedConv2d
        self.activation = torch.nn.ELU
        # self.activation = torch.nn.ReLU

        self.forget_bias = forget_bias
        self.flows = []
        self.param_list = []

        # For reordering z after each flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        for k in range(num_flows):
            arch_z = [ar_layer(z_size, h_size), self.activation()]
            self.param_list += list(arch_z[0].parameters())
            z_feats = torch.nn.Sequential(*arch_z)
            arch_zh = []
            for j in range(num_hidden):
                arch_zh += [ar_layer(h_size, h_size), self.activation()]
                self.param_list += list(arch_zh[-2].parameters())
            zh_feats = torch.nn.Sequential(*arch_zh)
            linear_mean = ar_layer(h_size, z_size, diagonal_zeros=True)
            linear_std = ar_layer(h_size, z_size, diagonal_zeros=True)
            self.param_list += list(linear_mean.parameters())
            self.param_list += list(linear_std.parameters())

            if torch.cuda.is_available():
                z_feats = z_feats.cuda()
                zh_feats = zh_feats.cuda()
                linear_mean = linear_mean.cuda()
                linear_std = linear_std.cuda()
            self.flows.append((z_feats, zh_feats, linear_mean, linear_std))

        self.param_list = torch.nn.ParameterList(self.param_list)

    def forward(self, z, h_context):

        logdets = 0.
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0 and not self.conv2d:
                # reverse ordering to help mixing
                z = z[:, self.flip_idx]

            h = flow[0](z)
            h = h + h_context
            h = flow[1](h)
            mean = flow[2](h)
            gate = torch.sigmoid(flow[3](h) + self.forget_bias)
            z = gate * z + (1 - gate) * mean
            logdets += torch.sum(gate.log().view(gate.size(0), -1), 1)
        return z, logdets


class CNN_FLOW_LAYER(nn.Module):
    def __init__(self, dim, kernel_size, dilation, rescale=True, skip=True):
        super(CNN_FLOW_LAYER, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.rescale = rescale
        self.skip = skip
        self.usecuda = True

        if self.rescale: # last layer of flow needs to account for the scale of target variable
            self.lmbd = nn.Parameter(torch.FloatTensor(self.dim).normal_().cuda())
        
        self.conv1d = nn.Conv1d(1, 1, kernel_size, dilation=dilation)
            
    def forward(self, x):
        # x is of size (bs x width)
        #kernel_width = 2
        #padding_len = (stride-1) * len + (kernel_size-1)

        # pad x periodically first, then do conv, this is a little complicated
        # padded_x = torch.cat((x, x[:, :(self.kernel_size-1)]), 1)

        # pad zero to the right
        padded_x = F.pad(x, (0, (self.kernel_size-1) * self.dilation))

        # tanh activation
        # activation = F.tanh(self.conv1d(
        # leaky relu activation

        conv1d = self.conv1d(
            padded_x.unsqueeze(1) # to make it (bs, 1, width)
        ).squeeze()

        w = self.conv1d.weight.squeeze()        

        # make sure u[i]w[0] >= -1 to ensure invertibility for h(x)=tanh(x) and with skip
        # tanh
        #activation = F.tanh(conv1d)
        #activation_gradient = 1 - activation**2

        neg_slope = 1e-2
        activation = F.leaky_relu(conv1d, negative_slope=neg_slope)
        activation_gradient = ((activation>=0).float() + (activation<0).float()*neg_slope)

        # for 0<=h'(x)<=1, ensure u*w[0]>-1
        scale = (w[0] == 0).float() * self.lmbd \
                +(w[0] > 0).float() * (-1./w[0] + F.softplus(self.lmbd)) \
                +(w[0] < 0).float() * (-1./w[0] - F.softplus(self.lmbd))


        '''
        activation = F.relu(self.conv1d(
            padded_x.unsqueeze(1) # to make it (bs, 1, width)
        ).squeeze())
        activation_gradient = (activation>=0).float()
        '''
        
        if self.rescale:
            output = activation.mm(torch.diag(scale))
            activation_gradient = activation_gradient.mm(torch.diag(scale))
        else:
            output = activation

        if self.skip:
            output = output + x

        # tanh'(x) = 1 - tanh^2(x)
        # leaky_relu'(x) = 1 if x >0 else 0.01
        # for leaky: leaky_relu_gradient = (output>0).float() + (output<0).float()*0.01
        # tanh
        # activation_gradient = (1 - activation**2).mm(torch.diag(self.lmbd))
        # leaky_relu


        if self.skip:
            logdet = torch.log(torch.abs(activation_gradient*w[0] + 1)).sum(1)
            #logdet = torch.log(torch.abs((activation_gradient*w[0]+1).prod(1) - (activation_gradient*w[1]).prod(1)))
        else:
            logdet = torch.log(torch.abs(activation_gradient*w[0])).sum(1)
            # logdet = torch.log(torch.abs(self.conv1d.weight.squeeze()[0]**self.dim - self.conv1d.weight.squeeze()[1]**self.dim)) + torch.log(torch.abs(nonlinear_gradient)).sum(1)

        return output, logdet
        

class DILATION_BLOCK(nn.Module):
    def __init__(self, dim, kernel_size):
        super(DILATION_BLOCK, self).__init__()

        self.block = nn.ModuleList()
        i = 0
        while 2**i <= dim:
            conv1d = CNN_FLOW_LAYER(dim, kernel_size, dilation=2**i)
            self.block.append(conv1d)
            i+= 1

    def forward(self, x):
        logdetSum = 0
        output = x
        for i in range(len(self.block)):
            output, logdet = self.block[i](output)
            logdetSum += logdet

        return output, logdetSum
        

class CNN_FLOW(nn.Module):
    def __init__(self, dim, cnn_layers, kernel_size, use_revert=True):
        super(CNN_FLOW, self).__init__()

        # prepare reversion matrix
        # just a matrix whose anti-diagonal are all 1s
        self.usecuda = True
        self.use_revert = use_revert
        self.R = Variable(torch.from_numpy(np.flip(np.eye(dim), axis=1).copy()).float(), requires_grad=False)
        if self.usecuda:
            self.R = self.R.cuda()
        
        self.layers = nn.ModuleList()
        for i in range(cnn_layers):
            #conv1d = CNN_FLOW_LAYER(kernel_size, kernel_size**i)  ## dilation setting
            #            block = CNN_FLOW_LAYER(dim, kernel_size, dilation=1)
            block = DILATION_BLOCK(dim, kernel_size)
            self.layers.append(block)
        
    def forward(self, x):
        logdetSum = 0
        output = x
        for i in range(len(self.layers)):
            output, logdet = self.layers[i](output)
            # revert the dimension of the output after each block
            if self.use_revert:
                output = output.mm(self.R)
            logdetSum += logdet

        return output, logdetSum
