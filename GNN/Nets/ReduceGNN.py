'''
RGNN here equals to the PharmGNN in paper
'''

from typing import Optional
# from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor
import torch, math
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Parameter, GRUCell
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class GATEConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 dropout: float = 0.0):
        super(GATEConv, self).__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class GNN(torch.nn.Module):

    def __init__(self, in_channels, channels,
                 out_channels, edge_dim, 
                 num_passing_atom=2, 
                 num_passing_pool=2,
                 num_passing_rg=2,
                 num_passing_mol=2,
                 dropout=0.0):
        super(GNN, self).__init__()

        self.num_passing_atom = num_passing_atom
        self.num_passing_pool = num_passing_pool
        self.num_passing_rg = num_passing_rg
        self.num_passing_mol = num_passing_mol
        self.dropout = dropout

        self.lin1 = Linear(in_channels, channels)

        conv = GATEConv(channels, channels, edge_dim, dropout)
        lin = Linear(channels,channels)
        res_lin = Linear(channels,channels)
        
        
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_lins = torch.nn.ModuleList([lin])
        self.atom_res_lins = torch.nn.ModuleList()

        for _ in range(num_passing_atom - 1):
            conv = GATConv(channels, channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_lins.append(lin)
            self.atom_res_lins.append(res_lin)  


        self.mol_conv = GATConv(channels, channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(channels, channels)

        self.lin2 = Linear(channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv, lin, res_lin in zip(self.atom_convs, self.atom_lins, self.atom_res_lins):
            conv.reset_parameters()
            lin.reset_parameters()
            res_lin.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, rg_data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x)) 
        res0 = F.leaky_relu_(self.atom_res_lins[0](x)) 
        m = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        m = F.dropout(m, p=self.dropout, training=self.training)
        x = (F.leaky_relu_(self.atom_lins[0](x)) + m).relu_()       

        for conv, lin, res_lin in zip(self.atom_convs[1:], self.atom_lins[1:], self.atom_res_lins[1:]):
            res = F.leaky_relu_(res_lin(x) )
            m = F.elu_(conv(x, edge_index))
            m = F.dropout(m, p=self.dropout, training=self.training)
            x = (F.leaky_relu_(lin(x)) + m + res0).relu_() 
            res0 = res

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_passing_mol):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        fp = F.dropout(out, p=self.dropout, training=self.training)
        out = (self.lin2(fp)).view(-1)
        return out, fp

class RGNN(torch.nn.Module):

    def __init__(self, in_channels, channels,
                 out_channels, edge_dim, 
                 num_passing_atom=2, 
                 num_passing_pool=2,
                 num_passing_rg=2,
                 num_passing_mol=2,
                 dropout=0.0):
        super(RGNN, self).__init__()

        self.num_passing_atom = num_passing_atom
        self.num_passing_pool = num_passing_pool
        self.num_passing_rg = num_passing_rg
        self.num_passing_mol = num_passing_mol
        self.dropout = dropout

        self.lin1 = Linear(in_channels, channels)

        # atom-level
        conv = GATEConv(channels, channels, edge_dim, dropout)
        lin = Linear(channels,channels)
        res_lin = Linear(channels,channels)
        
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_lins = torch.nn.ModuleList([lin])
        self.atom_res_lins = torch.nn.ModuleList()

        for _ in range(num_passing_atom - 1):
            conv = GATConv(channels, channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_lins.append(lin)
            self.atom_res_lins.append(res_lin)  

        # atom-RG
        self.pconv = GATConv(channels, channels, 
                                add_self_loops=False,
                                negative_slope=0.01)
        self.pgru = GRUCell(channels, channels)

        # RG-level
        self.lin_rg = torch.nn.Linear(channels+18, channels) 
        rg_econv = GATEConv(channels, channels, 1, dropout)
        rg_gru = GRUCell(channels, channels)
        self.rg_convs = torch.nn.ModuleList([rg_econv])
        self.rg_grus = torch.nn.ModuleList([rg_gru])
        for _ in range(num_passing_rg - 1):
            rg_conv = GATConv(channels, channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.rg_convs.append(rg_conv)
            self.rg_grus.append(GRUCell(channels, channels))


        # RG-molecule
        self.mol_conv = GATConv(channels, channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(channels, channels)

        self.lin2 = Linear(channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv, lin, res_lin in zip(self.atom_convs, self.atom_lins, self.atom_res_lins):
            conv.reset_parameters()
            lin.reset_parameters()
            res_lin.reset_parameters()
        self.pconv.reset_parameters()
        self.pgru.reset_parameters()
        self.lin_rg.reset_parameters()
        for rg_conv, rg_gru in zip(self.rg_convs, self.rg_grus):
            rg_conv.reset_parameters()
            rg_gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, rg_data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        rg_batch,rg_edge_index,rg_edge_attr = rg_data.batch, rg_data.edge_index, rg_data.edge_attr
        
        # atom-level
        x = F.leaky_relu_(self.lin1(x)) 
        res0 = F.leaky_relu_(self.atom_res_lins[0](x))
        m = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        m = F.dropout(m, p=self.dropout, training=self.training)
        x = (F.leaky_relu_(self.atom_lins[0](x)) + m).relu_()       

        for conv, lin, res_lin in zip(self.atom_convs[1:], self.atom_lins[1:], self.atom_res_lins[1:]):
            res = F.leaky_relu_(res_lin(x) )
            m = F.elu_(conv(x, edge_index))
            m = F.dropout(m, p=self.dropout, training=self.training)
            x = (F.leaky_relu_(lin(x)) + m + res0).relu_() 
            res0 = res
        
        # atom-RG 
        pool_index = data.pool_index.long()
        tid,sid = pool_index 
        mask = torch.unique(tid) # [R]
        pool_index = torch.stack([sid, tid],dim=0) 

        rg_out_ = scatter(x[sid], tid, dim=0, reduce='mean') 
        rg_out = rg_out_[mask] 

        for i in range(self.num_passing_pool):       
            h = F.elu_(self.pconv((x,rg_out_),pool_index)) # [R_,C]
            h = h[mask] # [R,C]
            rg_out = self.pgru(h, rg_out).relu_() # [R,C]
         
        # RG-level
        rg_out = torch.cat([rg_out,rg_data.x],dim=1) 
        rg_out = self.lin_rg(rg_out)

        h = F.elu_(self.rg_convs[0](rg_out, rg_edge_index, rg_edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        rg_out = self.rg_grus[0](h, rg_out).relu_()

        for rg_conv, rg_gru in zip(self.rg_convs[1:], self.rg_grus[1:]): 
            h = F.elu_(rg_conv(rg_out, rg_edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            rg_out = rg_gru(h, rg_out).relu_()


        # molecule readout
        row = torch.arange(rg_batch.size(0), device=batch.device)
        edge_index = torch.stack([row, rg_batch], dim=0)

        out = global_add_pool(rg_out, rg_batch).relu_()
        for t in range(self.num_passing_mol):
            h = F.elu_(self.mol_conv((rg_out, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        fp = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(fp).view(-1)
        return out,fp
