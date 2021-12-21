'''

run: nohup python MyNet_Classification.py --epochs 2 &
                                                                        
'''
import argparse
import os, os.path as osp
import numpy as np
import random
import torch
from torch_geometric.data import DataLoader
from torch import nn
import sys
sys.path.append('../.')  
sys.path.append('../../.')
from molecular_network.mol_dataset.reduceGraph_dataset import ReduceGraph_Dataset 
from molecular_network.mol_dataset.raw_dataset_4rgnn import Raw_Dataset
from configs.configs import * 
from Nets.NN import NN,ReduceNN
from Nets.AttentiveFP import AFP, ReduceAFP
from Nets.ReduceGNN import GNN, RGNN
from Utils.metrics import metrics_C
import time
torch.manual_seed(123)
start_time = str(time.ctime()).replace(':','-').replace(' ','_')

dirname, filename = os.path.split(os.path.abspath(__file__)) 
line = 'working folder>> ' + dirname + ' >> ' + filename + ' >> '
print(line)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, help='random seed')
parser.add_argument('--model', default='RGNN', help='')
parser.add_argument('--pretrain', default='', help='if True give checkpoints path')
parser.add_argument('--dataset', default='BBBP', help='')
parser.add_argument('--heads', default=2, help='')
parser.add_argument('--epochs', default=500, help='')
parser.add_argument('--x_latent_dim', default=256, help='')
parser.add_argument('--aggr', default='mean')
parser.add_argument('--dropout', default=0.0)
parser.add_argument('--cuda', default=1)
parser.add_argument('--checkpoints', default=False, action='store_true')
parser.add_argument('--choose_model', default='loss', choices=['loss','AUC'])
args = parser.parse_args()
torch.cuda.set_device(int(args.cuda))
seed = int(args.seed)
model_name = args.model
pretrain = args.pretrain
dataset_name = args.dataset   
epochs = int(args.epochs)
heads = int(args.heads)
x_dim = int(args.x_latent_dim)
aggr = args.aggr
dropout = float(args.dropout)
ckp = args.checkpoints
choose_model = args.choose_model
print('model parameters: ', vars(args))

# dataset
if dataset_name == 'Muta':
    option = Muta_train

if dataset_name == 'Muta_test':
    option = Muta_test

if dataset_name == 'Carcino':
    option = Carcino

if dataset_name == 'AMES':
    option = AMES

if dataset_name == 'HIV':
    option = HIV

if dataset_name == 'BACE':
    option = BACE

if dataset_name == 'BBBP':
    option = BBBP

if dataset_name == 'BioDeg':
    option = BioDeg

if dataset_name == 'ClinTox':
    option = ClinTox

if dataset_name == 'Tox21':
    option = Tox21 

if dataset_name == 'MUV':
    option = MUV

if dataset_name == 'SIDER':
    option = SIDER

if dataset_name == 'ToxCast':
    option = ToxCast

if dataset_name == 'AKT1':
    option = AKT1

if dataset_name == 'AURKA':
    option = AURKA

if dataset_name == 'BRAF':
    option = BRAF

if dataset_name == 'BTK':
    option = BTK

if dataset_name == 'CDK2':
    option = CDK2

if dataset_name == 'CK1':
    option = CK1

if dataset_name == 'EGFR':
    option = EGFR

if dataset_name == 'MAP4K2':
    option = MAP4K2

if dataset_name == 'mTOR':
    option = mTOR

if dataset_name == 'PIM1':
    option = PIM1

print('dataset option: ', option)  

# load dataset
dataset_option = {'root','raw_name','atom_types','bond_type_num','target_idxs'}
dataset_params = {key: value for key,value in option.items() if key in dataset_option}
dataset = Raw_Dataset(**dataset_params)
rg_dataset = ReduceGraph_Dataset(**dataset_params)
print('rg_dataset', rg_dataset)
print('dataset', dataset)

# read dataset dim
atom_dim = dataset[0].x.shape[1]
bond_dim = option['bond_type_num']
num_classes = option['num_classes']

# read model parameters
batch = option['batch']
num_passing= option['num_passing']
num_passing_before = option['num_passing_before']
num_passing_pool = option['num_passing_pool']
num_passing_after = option['num_passing_after']
num_passing_mol = option['num_passing_mol']


# training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
    
if model_name == 'NN':
    model = NN(x_in_dim=atom_dim, 
               edge_in_dim=bond_dim, 
               channels=x_dim, 
               num_passing=num_passing, 
               out_channels=num_classes
              ).to(device)

if model_name == 'ReduceNN':
    model = ReduceNN(x_in_dim=atom_dim, 
               edge_in_dim=bond_dim, 
               channels=x_dim, 
               num_passing_before=num_passing_before, 
               num_passing_after=num_passing_after,
               out_channels=num_classes
              ).to(device)


if model_name == 'AFP':
    model = AFP(in_channels=atom_dim,
                        hidden_channels=x_dim,
                        out_channels=num_classes, 
                        edge_dim=bond_dim, 
                        num_layers=num_passing,
                        num_timesteps=3,
                        dropout= 0.0
                        ).to(device)


if model_name == 'ReduceAFP':
    model = ReduceAFP(in_channels=atom_dim,
                        channels=x_dim,
                        out_channels=num_classes, 
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
                        out_channels=num_classes, 
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
                        out_channels=num_classes, 
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
pos_weight = option['pos_weight']
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=factor, patience=patience,
                                                        min_lr=0.00001)
criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor([float(pos_weight)])).to(device)


def train(epoch, dataloader, rg_dataloader):
    # --------* train *---------
    model.train()
    state_dict = model.state_dict()
    loss_all = 0
    y_trues = np.array([])
    y_preds = np.array([])
    y_probas = np.array([])

    for data,rg_data in zip(dataloader,rg_dataloader):
        data = data.to(device)
        rg_data = rg_data.to(device)
        optimizer.zero_grad()
        out = model(data,rg_data)
        y = data.y.view(-1)
        # print('y',y)
        mask =  ((data.y==1) | (data.y==0)).view(-1) 
        out = out[mask]
        y = y[mask]
        loss = criterion(out,y.float()).mean()
        loss.backward()
        loss_all += loss.item()
        y_proba = torch.sigmoid(out)
        y_pred = torch.round(y_proba)
        y_preds = np.concatenate((y_preds, y_pred.detach().cpu().numpy()), axis=0)
        y_probas = np.concatenate((y_probas, y_proba.detach().cpu().numpy()), axis=0) 
        y_trues = np.concatenate((y_trues, y.detach().cpu().numpy()), axis=0)
        optimizer.step()
    return state_dict, loss_all / len(dataloader.dataset), (y_trues, y_preds, y_probas)

    
def test(epoch, dataloader, rg_dataloader):
    # --------* test *---------
    model.eval()
    y_trues = np.array([])
    y_preds = np.array([])
    y_probas = np.array([])
    
    with torch.no_grad():
        for data,rg_data in zip(dataloader,rg_dataloader):
            data = data.to(device)
            rg_data = rg_data.to(device)
            out = model(data,rg_data)
            y = data.y.view(-1)
            mask =  ((data.y==1) | (data.y==0)).view(-1) 
            out = out[mask]
            y = y[mask]
            loss = criterion(out,y.float()).mean()
            y_proba = torch.sigmoid(out)
            y_pred = torch.round(y_proba)
            y_preds = np.concatenate((y_preds, y_pred.detach().cpu().numpy()), axis=0)
            y_probas = np.concatenate((y_probas, y_proba.detach().cpu().numpy()), axis=0) 
            y_trues = np.concatenate((y_trues, y.detach().cpu().numpy()), axis=0)



    return loss, (y_trues, y_preds, y_probas)

    
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
    ckp_folder = osp.join(dirname, 'checkpoints', dataset_name) 
    os.makedirs(ckp_folder) if not os.path.exists(ckp_folder) else None   


# random shuffle two datasets 
random.seed(seed)
index = [i for i in range(len(dataset))]
random.shuffle(index)
dataset = dataset[index]
rg_dataset = rg_dataset[index]
print('top five items', index[:5])

split = len(dataset) // 10
dataset.data.y = dataset.data.y.long()
train_dataset = dataset[:-2*split]
val_dataloader = DataLoader(dataset[-2*split:-split], batch_size=batch) # 10%
test_dataloader = DataLoader(dataset[-split:], batch_size=batch) # 10%
train_rg_dataset = rg_dataset[:-2*split]

val_rg_dataloader = DataLoader(rg_dataset[-2*split:-split], batch_size=batch) # 10%
test_rg_dataloader = DataLoader(rg_dataset[-split:], batch_size=batch) # 10%

best_params = {}
best_params['val_AUC'] = None
best_params['val_loss'] = None
best_params['epoch_AUC'] = None
best_params['epoch_loss'] = None

for epoch in range(epochs):                 
    lr = scheduler.optimizer.param_groups[0]['lr']

    random.seed(epoch)
    index = [i for i in range(len(train_dataset))]
    random.shuffle(index)
    train_dataset = train_dataset[index]
    train_rg_dataset = train_rg_dataset[index]
    train_dataloader = DataLoader(train_dataset, batch_size=batch) # 80%
    train_rg_dataloader = DataLoader(train_rg_dataset, batch_size=batch) # 80%  
    
    state_dict, loss, train_pair = train(epoch, train_dataloader, train_rg_dataloader) #train_pair 代表 y_true, y_prds对
    val_loss, val_pair = test(epoch, val_dataloader, val_rg_dataloader)
    scheduler.step(val_loss)

    # choose model based on AUC of validation dataset
    val_AUC = metrics_C(*val_pair)['AUC']
    train_AUC = metrics_C(*train_pair)['AUC']

    if best_params['val_AUC'] is None or val_AUC >= best_params['val_AUC']:
        # _, valid_pair = test(epoch, val_dataloader, val_rg_dataloader)
        # _, test_pair = test(epoch, test_dataloader, test_rg_dataloader)
        
        best_params['val_AUC'] = val_AUC
        best_params['epoch_AUC'] = epoch 
        best_model = state_dict

        if choose_model == 'AUC':
            # save model performance with the highest AUC of validation dataset
            _, test_pair = test(epoch, test_dataloader, test_rg_dataloader)
            test_AUC = metrics_C(*test_pair)['AUC']
            metrics_train = metrics_C(*train_pair)
            metrics_val = metrics_C(*val_pair)
            metrics_test = metrics_C(*test_pair)
        
    if best_params['val_loss'] is None or val_loss <= best_params['val_loss']:
        best_params['val_loss'] = val_loss
        best_params['epoch_loss'] = epoch

        if choose_model == 'loss':
            _, test_pair = test(epoch, test_dataloader, test_rg_dataloader)
            test_AUC = metrics_C(*test_pair)['AUC']
            # save model performance with the lowest loss of validation dataset
            metrics_train = metrics_C(*train_pair)
            metrics_val = metrics_C(*val_pair)
            metrics_test = metrics_C(*test_pair)
        

    if (epoch - best_params['epoch_AUC'] > 20) and (epoch - best_params['epoch_loss'] > 15):
        break

    print('Epoch: {:03d}, LR: {:6f}, Loss: {:.6f},\
           Train AUC: {:.3f}, Valid AUC: {:.3f}, \
           Test AUC: {:.3f}, Valid loss:{:.6f}'.format(
                            epoch, 
                            lr, 
                            loss, 
                            train_AUC, 
                            val_AUC, 
                            test_AUC,
                            val_loss)
                            )   

# save model
if ckp:
    ckp_path =  osp.join(ckp_folder, model_name 
                            + 'epoch_' + str(best_params['epoch_AUC'])
                            + '_' + start_time + '.ckp')
    torch.save(best_model, ckp_path)

print('best_params', best_params)
print('final training set metrics: ', metrics_train)
print('final valid set metrics: ', metrics_val)
print('final test set metrics: ', metrics_test)