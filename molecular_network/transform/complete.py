import sys
sys.path.append('/home/tuotuoyue/private-kongyue/MN')
import torch
import torch_geometric.transforms as T

class Complete_Virtual_Self(object):

    def __call__(self, data):


        row = torch.arange(data.num_nodes, dtype=torch.long)
        col = torch.arange(data.num_nodes, dtype=torch.long)


        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            real_full_edge_idx = torch.arange(data.num_nodes * data.num_nodes, dtype=torch.long
            
            real_edge_idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]

            real_edge_self_idx = torch.arange(data.num_nodes, dtype=torch.long) * (data.num_nodes+1)

            vitural_edge_idx = list(set(real_full_edge_idx.numpy().tolist()) - set(real_edge_idx.numpy().tolist()) - set(real_edge_self_idx.numpy().tolist()))

            
            size = [data.edge_attr.size()[0],6] 
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            

            edge_attr[real_edge_idx, :4] = data.edge_attr 
            edge_attr[real_edge_self_idx, 5] = 1
            edge_attr[vitural_edge_idx, 4] = 1

        data.edge_attr = edge_attr

        data.edge_index = edge_index


        return data

class Complete_Virtual(object):

    def __call__(self, data):


        row = torch.arange(data.num_nodes, dtype=torch.long)
        col = torch.arange(data.num_nodes, dtype=torch.long)


        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)


        edge_attr = None
        if data.edge_attr is not None:

            real_full_edge_idx = torch.arange(data.num_nodes * data.num_nodes, dtype=torch.long)
            real_edge_idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]


            

            vitural_edge_idx = list(set(real_full_edge_idx.numpy().tolist()) - set(real_edge_idx.numpy().tolist()))
          
            size = [data.edge_attr.size()[0],5] 
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            

            edge_attr[real_edge_idx, :4] = data.edge_attr 
            edge_attr[vitural_edge_idx, 4] = 1

        data.edge_attr = edge_attr
        data.edge_index = edge_index


        return data


class Complete_Fake(object):
    def __init__(self,n_max=6):
        self.n_max=n_max
    
    def __call__(self, data):


        row = torch.arange(self.n_max, dtype=torch.long)
        col = torch.arange(self.n_max, dtype=torch.long)


        row = row.view(-1, 1).repeat(1, self.n_max).view(-1)
        col = col.repeat(self.n_max)
        edge_index = torch.stack([row, col], dim=0)


        edge_attr = None
        if data.edge_attr is not None:

            fake_full_edge_idx = torch.arange(self.n_max * self.n_max, dtype=torch.long)

            real_edge_idx = data.edge_index[0] * self.n_max + data.edge_index[1]

            fake_edge_idx = list(set(fake_full_edge_idx.numpy().tolist())-set(real_edge_idx.numpy().tolist()))

            real_edge_size = data.edge_attr.size()
            fake_edge_size = (self.n_max * self.n_max, real_edge_size[1]+1)
            edge_attr = data.edge_attr.new_zeros(fake_edge_size)   
            edge_attr[real_edge_idx,:-1] = data.edge_attr        
            edge_attr[fake_edge_idx, -1] = 1
            

            real_x_size = data.x.size()
            fake_x_size = (self.n_max, real_x_size[1]+1)
            x = data.x.new_zeros(fake_x_size)
            x[:real_x_size[0], :real_x_size[1]] = data.x
            x[real_x_size[0]:,-1] = 1

        data.edge_attr = edge_attr
        data.edge_index = edge_index
        data.x = x


        return data