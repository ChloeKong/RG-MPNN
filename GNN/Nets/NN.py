'''
NN is short for MPNN
'''
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, GRUCell
from torch_scatter import scatter
from torch_geometric.nn import NNConv, Set2Set, GATConv

    

class NN(torch.nn.Module):
    '''
    nets for regression task
    '''
    def __init__(self, x_in_dim=28, edge_in_dim=4, channels=64, out_channels=1, num_passing=4, aggr='mean'):
        super(NN, self).__init__()
        self.lin0 = torch.nn.Linear(x_in_dim, channels)
        self.num_passing = num_passing

        nn = Sequential(Linear(edge_in_dim, 128), ReLU(), Linear(128, channels * channels))
        self.conv = NNConv(channels, channels, nn, aggr=aggr)
        self.gru = GRU(channels, channels)

        self.set2set = Set2Set(channels, processing_steps=num_passing)
        self.lin1 = torch.nn.Linear(2 * channels, channels)
        self.lin2 = torch.nn.Linear(channels, out_channels)

    def forward(self, data, rg_data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.num_passing):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        fp = F.relu(self.lin1(out))
        out = self.lin2(fp)
        return out.view(-1),fp


class ReduceNN(torch.nn.Module):
    '''
    nets for regression task
    '''
    def __init__(self, x_in_dim=28, edge_in_dim=4, channels=64, out_channels=1, num_passing_before=2, num_passing_pool =2, num_passing_after=2,aggr='mean'):
        super(ReduceNN, self).__init__()
        self.lin0 = torch.nn.Linear(x_in_dim, channels)
        self.num_passing_before = num_passing_before
        self.num_passing_pool = num_passing_pool
        self.num_passing_after = num_passing_after

        nn1 = Sequential(Linear(edge_in_dim, 128), ReLU(), Linear(128, channels * channels))
        self.conv1 = NNConv(channels, channels, nn1, aggr=aggr)
        self.gru1 = GRU(channels, channels)

        self.pconv = GATConv(channels, channels, 
                                add_self_loops=False,
                                negative_slope=0.01)
        self.pgru = GRUCell(channels, channels)

        nn2 = Sequential(Linear(1, 64), ReLU(), Linear(64, channels * channels))
        self.conv2 = NNConv(channels, channels, nn2, aggr=aggr)
        self.gru2 = GRU(channels, channels)

        self.lin_rg = torch.nn.Linear(channels+18, channels) 

        self.set2set = Set2Set(channels, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * channels, channels)
        self.lin2 = torch.nn.Linear(channels, out_channels) #

    def forward(self, data, rg_data):
        pool_index = data.pool_index.long()
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        # atom-level
        for i in range(self.num_passing_before):
            m = F.relu(self.conv1(out, data.edge_index, data.edge_attr))
            out, h = self.gru1(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # atom-RG
        tid,sid = pool_index 
        mask = torch.unique(tid) # [R]
        pool_index = torch.stack([sid, tid],dim=0) 

        rg_out_ = scatter(out[sid], tid, dim=0, reduce='add') 
        rg_out = rg_out_[mask]

        for i in range(self.num_passing_pool):        
            h = F.elu_(self.pconv((out,rg_out_),pool_index)) # [R_,C]
            h = h[mask] # [R,C]
            rg_out = self.pgru(h, rg_out).relu_() # [R,C]

        # RG-level
        rg_out = torch.cat([rg_out,rg_data.x],dim=1) 
        rg_out = self.lin_rg(rg_out)
        h = rg_out.unsqueeze(0)
        
        for i in range(self.num_passing_after):
            m = F.relu(self.conv2(rg_out, rg_data.edge_index, rg_data.edge_attr))
            rg_out, h = self.gru2(m.unsqueeze(0), h)
            rg_out = rg_out.squeeze(0)  # [R,C]
       
        # molecule readout
        out = self.set2set(rg_out, rg_data.batch) #[B,2C]

        fp = F.relu(self.lin1(out)) # [B,C]
        out = self.lin2(fp) # [B,1]
        return out.view(-1),fp