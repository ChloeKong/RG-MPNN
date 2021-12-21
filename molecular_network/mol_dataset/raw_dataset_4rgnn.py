#%%
import sys
sys.path.append('../..')  
print(sys.path)
import os
import os.path as osp
import pandas as pd
import numpy as np
import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from molecular_network.mol_feature.atom_feature import  atom_feature
from molecular_network.mol_feature.bond_feature import bond_feature
from molecular_network.mol_feature.atom_feature import atom_types
from molecular_network.mol_feature.reduceGraph_feature import rg_feature, reduce_graph_to_mol, rg_x_feature
from molecular_network.transform.reduce_graph import reduceGraph
import rdkit
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from molecular_network.transform.complete import Complete_Virtual, Complete_Fake
from molecular_network.util.wash import NeutraliseCharges




class Raw_Dataset(InMemoryDataset):

    def __init__(self, root, raw_name, target_idxs=False, info_idxs=False, 
                atom_types=atom_types,hybrid=True, aromatic=True, 
                exp_val=True, Hs=True, bond_type_num=4,
                transform=None, pre_transform=None, pre_filter=None, num_classes=None):
        self.raw_name = root + '/raw/' + raw_name
        self.processed_name = root  + '/processed/' + raw_name[:-4] + '_raw_data.pt'
        self.rg_smi_name = root  + '/processed/' + raw_name[:-4] + '_rg_smiles.csv'
        self.info_idxs = info_idxs
        self.target_idxs = target_idxs
        self.atom_types = atom_types
        self.hybrid = hybrid
        self.aromatic = aromatic
        self.exp_val = exp_val
        self.Hs = Hs
        self.bond_type_num = bond_type_num
        super(Raw_Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_name) 
    

    @property
    def raw_file_names(self):
        return self.raw_name

    @property
    def processed_file_names(self):
        return self.processed_name
        


    def process(self):

        if '.csv' in self.raw_paths[0]:
            df = pd.read_csv(self.raw_paths[0])
            s = df.iloc[:,0] 
        
        if self.target_idxs:
            if self.target_idxs == 1:
                self.target_idxs = [1]
                
            target = df.iloc[:,self.target_idxs]
            target_names = df.columns[self.target_idxs].tolist()

        if self.info_idxs:
            df_info = df.iloc[:,self.info_idxs]


        data_list = []
        remover = SaltRemover()
        rg_smi_list = []
        valid_smi_list = []
        y_list = []

        for i,smi in enumerate(s):

            if i %100 == 0:
                print('process %i molecules' %(i))             

            try:
                mol = Chem.MolFromSmiles(smi)
                atoms_num=mol.GetNumAtoms()
            except:
                print('rdkit reading failed',smi)
                continue
            error_cmpd_count = 0

            if '%' in smi:
                print('\% molecule: ', smi)
                continue # 


            try:
                mol = remover.StripMol(mol,dontRemoveEverything=True)
                mol = NeutraliseCharges(mol)
                x = atom_feature(mol,atom_types=self.atom_types,hybrid=self.hybrid, aromatic=self.aromatic, exp_val=self.exp_val, Hs=self.Hs)
            except: 
                error_cmpd_count += 1
                print('error_compound: ', smi)
                continue


            try:
                edge_index, edge_attr = bond_feature(mol,bond_type_num=self.bond_type_num)  #,fully_connect=self.fc
            except:
                print('skip mol with one atom',smi)
                continue

            try:
                rg_name, pool_index, rg_edge_index_attr = rg_feature(mol,edge_index) 
                pool_index = pool_index.transpose(0,1)
                rg_x = rg_x_feature(rg_name)
            except:
                print('skip mols with long C chain',smi)
                continue
            
            try:
                rg_edge_index = rg_edge_index_attr.transpose(0,1)[:-1,:]
                rg_edge_attr = rg_edge_index_attr[:,[2]]
            except:
                print('skip mol with one RG',smi)
                continue 

            rg_mol = reduce_graph_to_mol(rg_name, rg_edge_index_attr)
            rg_smi = Chem.MolToSmiles(rg_mol)
            # rg_molblock = Chem.MolToMolBlock(rg_mol)

            if '.' in rg_smi:
                continue
            if '%' in rg_smi:
                continue # 
        


            if self.target_idxs:
                y_ = target.iloc[i,:].to_list()                
                y = torch.tensor(y_, dtype=torch.float)#.unsqueeze(0)
            else:
                y = None

            y_list.append(y_)
            rg_smi_list.append(rg_smi)
            valid_smi_list.append(smi)

            if self.info_idxs:
                info = df_info.loc[i]
            else:
                info = None

            data = Data(x=x, 
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pool_index=pool_index,
                        rg_edge_index=rg_edge_index,
                        rg_edge_attr=rg_edge_attr,               
                        y=y, 
                        info=info)
                        
            data_list.append(data)


        torch.save(self.collate(data_list), self.processed_name)
        

        rg = pd.DataFrame(rg_smi_list)
        smi = pd.DataFrame(valid_smi_list)
        label = pd.DataFrame(y_list)

        df = pd.concat([rg, smi, label], axis=1)
        df.columns = ['rg','smi'] + target_names#['label']*label_num
        df.to_csv(self.rg_smi_name,index=False)        
        

