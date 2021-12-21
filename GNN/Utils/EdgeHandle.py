'''

    EdgeHandle
                                                                         
'''
from torch.nn.functional import *
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch_geometric.nn.inits import reset, uniform, zeros
from torch_geometric.nn import NNConv, GCNConv, radius_graph, VGAE
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d as BN, LeakyReLU
from torch import nn

        
def self_edge_attr(out, edge_index, edge_attr): 

    e = torch.arange(0, out.shape[0])
    sid, tid = edge_index
    self_mask = torch.eq(sid, tid)

    #返回自身对应的[2, E]和边特征
    return torch.stack([e, e], dim=0), edge_attr[self_mask]

def neighbour_edge_attr(out, edge_index, edge_attr): 
    
    sid, tid = edge_index
    self_mask = torch.eq(sid, tid)
    mask2 = self_mask == False
    
    #返回对应邻居节点的[2, E]和边特征
    return edge_index[:, mask2], edge_attr[mask2]
