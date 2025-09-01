# Modified from: https://github.com/microsoft/AI2BMD/blob/ViSNet/visnet/models/utils.py
from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing


class CosineCutoff(nn.Module):
    
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=False):
        super(GaussianSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(0, self.cutoff, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


act_func = {
    "ssp": ShiftedSoftplus, 
    "silu": nn.SiLU, 
    "tanh": nn.Tanh, 
    "sigmoid": nn.Sigmoid, 
    "swish": Swish, 
    'leakyrelu': nn.LeakyReLU(), 
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'elu': nn.ELU,
    'softmax': nn.Softmax,
    'selu': nn.SELU,
}


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.eps = 1e-12
        
        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)
        
        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm
        
        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)
    
    def none_norm(self, vec):
        return vec
        
    def rms_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist ** 2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)
    
    def max_min_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1, keepdim=True)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        direct = vec / dist
        
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        
        return F.relu(dist) * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")


class NeighborEmbedding(MessagePassing):
    def __init__(self, input_channels, hidden_channels, num_dist_rbf_intra, num_dist_rbf_inter, cutoff_intra, cutoff_inter):
        super(NeighborEmbedding, self).__init__(aggr="add")
        
        self.lin_t = nn.Sequential(nn.Linear(input_channels, hidden_channels), nn.SiLU())
        
        self.s_proj = nn.Linear(input_channels, hidden_channels)
        self.t_proj = nn.Linear(input_channels, hidden_channels)
        
        self.dist_t_proj = nn.Linear(num_dist_rbf_intra, hidden_channels)
        self.dist_s2t_proj = nn.Linear(num_dist_rbf_inter, hidden_channels)
        
        self.trans_inter = nn.Linear(hidden_channels, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 3, hidden_channels)
        self.cutoff_intra = CosineCutoff(cutoff_intra)
        self.cutoff_inter = CosineCutoff(cutoff_inter)
        
        #self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.dist_t_proj.weight)
        nn.init.xavier_uniform_(self.dist_s2t_proj.weight)
        nn.init.xavier_uniform_(self.trans_inter.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.dist_t_proj.bias.data.fill_(0)
        self.dist_s2t_proj.bias.data.fill_(0)
        self.trans_inter.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, x_s, x_t, edge_index_t, edge_index_s2t, dist_t, dist_s2t, rbf_dist_t, rbf_dist_s2t):
        
        x = self.lin_t(x_t)
        
        C_t = self.cutoff_intra(dist_t)
        W_t = self.dist_t_proj(rbf_dist_t) * C_t.view(-1, 1)

        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors_t = self.propagate(edge_index_t, x=self.t_proj(x_t), W=W_t, size=None)
        
        C_s2t = self.cutoff_inter(dist_s2t)
        W_s2t = self.dist_s2t_proj(rbf_dist_s2t) * C_s2t.view(-1, 1)
        
        x_neighbors_s2t = self.propagate(edge_index_s2t, x=self.s_proj(x_s), W=W_s2t, size=(x_s.size(0), x_t.size(0)))
        x_neighbors_s2t = self.trans_inter(x_neighbors_s2t)
        
        out = self.combine(torch.cat([x, x_neighbors_t, x_neighbors_s2t], dim=1))
        #out = x + self.combine(torch.cat([x_neighbors_t, x_neighbors_s2t], dim=1))
        
        return out

    def message(self, x_j, W):
        return x_j * W

    
class EdgeEmbedding(MessagePassing):
    
    def __init__(self, num_rbf, hidden_channels, edge_type='intra'):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels)
        
        if edge_type=='inter':
            self.cat_proj = nn.Linear(hidden_channels * 2, hidden_channels)
        else:
            self.register_parameter("cat_proj", None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        
    def forward(self, edge_index, dist_rbf, x_s, x_t):
        # propagate_type: (x: PairTensor, edge_attr: Tensor)
        
        out = self.propagate(edge_index, x_s=x_s, x_t=x_t, rbf=dist_rbf)
        return out
    
    def message(self, x_s_j, x_t_i, rbf):
        rbf = self.rbf_proj(rbf) 
        return self.cat_proj(torch.cat([x_s_j * rbf, x_t_i * rbf], 1)) if self.cat_proj is not None else (x_s_j + x_t_i) * rbf 
    
    def aggregate(self, features, index):
        # no aggregate
        return features
        

class NeighborEmbeddingHomo(MessagePassing):
    def __init__(self, input_channels, hidden_channels, num_dist_rbf, cutoff):
        super(NeighborEmbeddingHomo, self).__init__(aggr="add")
        
        self.lin = nn.Sequential(nn.Linear(input_channels, hidden_channels), nn.SiLU())
        
        self.proj = nn.Linear(input_channels, hidden_channels)
        
        self.dist_proj = nn.Linear(num_dist_rbf, hidden_channels)

        self.cutoff = CosineCutoff(cutoff)


    def forward(self, x, edge_index, dist, rbf_dist):
        
        out = self.lin(x)
        
        C = self.cutoff(dist)
        W = self.dist_proj(rbf_dist) * C.view(-1, 1)

        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=self.proj(x), W=W, size=None)
  
        out = out + x_neighbors
        return out

    def message(self, x_j, W):
        return x_j * W
