'''

run: nohup python MyNet_Regression.py --epochs 2 &
                                                                        
'''
import random
import argparse
import os, os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch import nn
import sys
sys.path.append('../.')  
sys.path.append('../../.')
from molecular_network.mol_dataset.reduceGraph_dataset import ReduceGraph_Dataset 
from molecular_network.mol_dataset.raw_dataset_4rgnn import Raw_Dataset
from configs.configs import * 
from Nets.NN import NN, ReduceNN
from Nets.AttentiveFP import AFP, ReduceAFP
from Nets.ReduceGNN import GNN, RGNN
from Utils.metrics import metrics_R
torch.manual_seed(123)

dirname, filename = os.path.split(os.path.abspath(__file__)) 
print('working folder>> ',dirname, ' >> ', filename, ' >> ')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, help='random seed')
parser.add_argument('--model', default='RGNN', help='')
parser.add_argument('--pretrain', default='', help='if True give checkpoints path')
parser.add_argument('--dataset', default='Lipo', help='one of [Lipo, Muta, Muta_test]')
parser.add_argument('--batch', default=64, help='')
parser.add_argument('--heads', default=2, help='')
parser.add_argument('--epochs', default=200, help='')
parser.add_argument('--x_latent_dim', default=256, help='')
parser.add_argument('--num_passing', default=3)
parser.add_argument('--num_passing_before', default=3)
parser.add_argument('--num_passing_pool', default=1)
parser.add_argument('--num_passing_after', default=1)
parser.add_argument('--num_passing_mol', default=1)
parser.add_argument('--aggr', default='mean')
parser.add_argument('--dropout', default=0.0)
parser.add_argument('--cuda', default=1)
parser.add_argument('--checkpoints', default=False, action='store_true')
args = parser.parse_args()
torch.cuda.set_device(int(args.cuda))
seed = int(args.seed)
model_name = args.model
pretrain = args.pretrain
dataset = args.dataset    
batch = int(args.batch)
epochs = int(args.epochs)
heads = int(args.heads)
x_dim = int(args.x_latent_dim)
num_passing= int(args.num_passing)
num_passing_before = int(args.num_passing_before)
num_passing_pool = int(args.num_passing_pool)
num_passing_after = int(args.num_passing_after)
num_passing_mol = int(args.num_passing_mol)
aggr = args.aggr
dropout = float(args.dropout)
ckp = args.checkpoints
print('model parameters: ', vars(args))

# dataset
if dataset == 'Lipo':
    option = Lipo

if dataset == 'FreeSolv':
    option = FreeSolv

if dataset == 'ESOL':
    option = ESOL
    

print('数据集参数: ', option)   

# load dataset
dataset_option = {'root','raw_name','atom_types','bond_type_num','target_idxs'}
dataset_params = {key: value for key,value in option.items() if key in dataset_option}
rg_dataset = ReduceGraph_Dataset(**dataset_params)
dataset = Raw_Dataset(**dataset_params)

# read dataset dim
atom_dim = dataset[0].x.shape[1]
bond_dim = option['bond_type_num']

# model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

    
if model_name == 'NN':
    model = NN(x_in_dim=atom_dim, 
               edge_in_dim=bond_dim, 
               channels=x_dim, 
               num_passing=num_passing,
               aggr=aggr
              ).to(device)

    
if model_name == 'ReduceNN':
    model = ReduceNN(
               x_in_dim=atom_dim, 
               edge_in_dim=bond_dim, 
               channels=x_dim, 
               num_passing_before= 3, 
               num_passing_pool = 1, 
               num_passing_after= 1,
               aggr=aggr
              ).to(device)


if model_name == 'AFP':
    model = AFP(in_channels=atom_dim,
                        hidden_channels=x_dim,
                        out_channels=1, 
                        edge_dim=bond_dim, 
                        num_layers=num_passing,
                        num_timesteps=3,
                        dropout= 0.0
                        ).to(device)


if model_name == 'ReduceAFP':
    model = ReduceAFP(in_channels=atom_dim,
                        channels=x_dim,
                        out_channels=1, 
                        edge_dim=bond_dim, 
                        num_passing_atom=num_passing_before,
                        num_passing_pool=num_passing_pool,
                        num_passing_rg=num_passing_after,
                        num_passing_mol=num_passing_mol,
                        dropout= 0.0
                        ).to(device)

if model_name == 'GNN':
    model = GNN(in_channels=atom_dim,
                        channels=x_dim,
                        out_channels=1, 
                        edge_dim=bond_dim, 
                        num_passing_atom=num_passing_before,
                        num_passing_pool=num_passing_pool,
                        num_passing_rg=num_passing_after,
                        num_passing_mol=num_passing_mol,
                        dropout= 0.0
                        ).to(device)

if model_name == 'RGNN':
    model = RGNN(in_channels=atom_dim,
                        channels=x_dim,
                        out_channels=1, 
                        edge_dim=bond_dim, 
                        num_passing_atom=num_passing_before,
                        num_passing_pool=num_passing_pool,
                        num_passing_rg=num_passing_after,
                        num_passing_mol=num_passing_mol,
                        dropout= dropout
                        ).to(device)



base_lr = option['base_lr']
factor = option['factor']
patience = option['patience']

optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=factor, patience=patience, 
                                                        min_lr=0.00001) 


def train(epoch, dataloader, rg_dataloader):
    # --------* train *---------
    model.train()
    state_dict = model.state_dict()
    loss_all = 0
    y_true = np.array([])
    y_prds = np.array([])

    for data, rg_data in zip(dataloader,rg_dataloader):
        data = data.to(device)
        rg_data = rg_data.to(device)
        y = data.y.view(-1)
        optimizer.zero_grad()
        out, fp = model(data,rg_data)
        loss = F.mse_loss(out, y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs 
        y_prds = np.concatenate((y_prds, (out*std).detach().cpu().numpy()), axis=0)
        y_true = np.concatenate((y_true, (y*std).detach().cpu().numpy()), axis=0)
        optimizer.step()

    return state_dict, loss_all / len(dataloader.dataset), (y_true, y_prds)

    
def test(epoch, dataloader, rg_dataloader):
    model.eval()
    error = 0
    y_true = np.array([])
    y_prds = np.array([])
    
    with torch.no_grad():
        for data,rg_data in zip(dataloader,rg_dataloader):
            data = data.to(device)
            rg_data = rg_data.to(device)
            y = data.y.view(-1)
            out, fp = model(data,rg_data)
            error += (out * std - y * std).abs().sum().item()  # MAE
            y_prds = np.concatenate((y_prds, (out*std).cpu().numpy()), axis=0)
            y_true = np.concatenate((y_true, (y*std).detach().cpu().numpy()), axis=0)

    mae = error / len(dataloader.dataset)
    return mae, (y_true, y_prds)

# if need, load checkpoint and then continue train
if pretrain:
    print(model)
    state_dict = model.state_dict()
    model.load_state_dict(torch.load(pretrain))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


# save checkpoint files
if ckp:
    ckp_folder = osp.join(dirname, '..', 'checkpoints', filename[:-3]) 
    os.makedirs(ckp_folder) if not os.path.exists(ckp_folder) else None   

dataset.data.y = torch.Tensor(dataset.data.y)
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.item(), std.item()

# random shuffle two datasets 
random.seed(seed)
index = [i for i in range(len(dataset))]
random.shuffle(index)
dataset = dataset[index]
rg_dataset = rg_dataset[index]
print('top five items', index[:5])

split = len(dataset) // 10
train_dataloader = DataLoader(dataset[:-2*split], batch_size=batch) # 80%
val_dataloader = DataLoader(dataset[-2*split:-split], batch_size=batch) # 10%
test_dataloader = DataLoader(dataset[-split:], batch_size=batch) # 10%
train_rg_dataloader = DataLoader(rg_dataset[:-2*split], batch_size=batch) # 80%
val_rg_dataloader = DataLoader(rg_dataset[-2*split:-split], batch_size=batch) # 10%
test_rg_dataloader = DataLoader(rg_dataset[-split:], batch_size=batch) # 10%



best_val_error = None
for epoch in range(epochs):    
                
    lr = scheduler.optimizer.param_groups[0]['lr']
    state_dict, loss, train_pair = train(epoch, train_dataloader, train_rg_dataloader) #train_pair 代表 y_true, y_prds对
    val_mae, val_pair = test(epoch, val_dataloader, val_rg_dataloader)
    scheduler.step(val_mae)

    if best_val_error is None or val_mae <= best_val_error:
        test_mae, test_pair = test(epoch, test_dataloader, test_rg_dataloader)
        best_val_error = val_mae
    
    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
        'Test MAE: {:.7f}'.format(epoch, lr, loss, val_mae, test_mae))
    
    metrics_train = metrics_R(*train_pair)
    metrics_val = metrics_R(*val_pair)
    metrics_test = metrics_R(*test_pair)
    print('metrics_test: ', metrics_test)


    # save model
    if ckp:
        ckp_path =  osp.join(ckp_folder, str(epoch) + '.ckp')
        torch.save(state_dict, ckp_path)

print('final training set metrics: ', metrics_train)
print('final valid set metrics: ', metrics_val)
print('final test set metrics: ', metrics_test)