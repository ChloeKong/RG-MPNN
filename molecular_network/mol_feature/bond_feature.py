#!/usr/bin/env python
# coding: utf-8



import numpy as np

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDConfig
from rdkit.Chem.rdchem import BondType as BT
rdBase.DisableLog('rdApp.error')


# from ..util import onehot, ringSize_a


bool_dict = {False: 0, True: 1} 
# bond_5types = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3, 'virtual_bond': 4}
bond_4types = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}
bond_3types = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2}
stereo_type_dict = {'STEREOZ': 0, 'STEREOE': 1,'STEREOANY': 2, 'STEREONONE': 3}
in_ring_dict = bool_dict
is_conjugate_dict = bool_dict

def bond_feature(mol,
                 fully_connect=False,
                 bond_type_num=3,
                 stereo_type_dict=stereo_type_dict,
                 in_ring_dict=in_ring_dict,
                 is_conjugate_dict=is_conjugate_dict,
                 stereo=False,
                 in_ring=False,
                 is_conjugate=False
                 ):
                 


    N = mol.GetNumAtoms()

    row, col, bond_idx, stereo_idx, in_ring_idx, is_conjugate_idx = [], [], [], [], [], []
    
    if bond_type_num == 3:
        bond_type_dict = bond_3types
        Chem.Kekulize(mol, clearAromaticFlags=True)
    else: 
        bond_type_dict = bond_4types

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]

        bond_idx += 2 * [bond_type_dict[str(bond.GetBondType())]]
        if stereo == True:
            stereo_idx += 2 * [stereo_type_dict[str(bond.GetStereo())]]
        if in_ring == True:
            in_ring_idx += 2 * [in_ring_dict[bond.IsInRing()]]
        if is_conjugate == True:
            is_conjugate_idx += 2 * [is_conjugate_dict[bond.GetIsConjugated()]]

    if fully_connect == True:
        
        adj = Chem.GetAdjacencyMatrix(mol)
        virtual_edge_index = [i.tolist() for i in np.where(adj==0)]
        length = len(virtual_edge_index[0])
        virtual_idx= [4] * length 

        row = row + virtual_edge_index[0]
        col = col + virtual_edge_index[1]
        bond_idx = bond_idx + virtual_idx

    edge_index = torch.tensor([row, col], dtype=torch.long)

    _ = []
    edge_attr_bond_type = F.one_hot(
       torch.tensor(bond_idx),
       num_classes=len(bond_type_dict)).to(torch.float)
    _.append(edge_attr_bond_type)
    if stereo == True:
        edge_attr_stereo_type = F.one_hot(
           torch.tensor(stereo_idx),
           num_classes=len(stereo_type_dict)).to(torch.float)
        _.append(edge_attr_stereo_type)
    if in_ring == True:
        edge_attr_in_ring_type = F.one_hot(
           torch.tensor(in_ring_idx),
           num_classes=len(in_ring_dict)).to(torch.float)
        _.append(edge_attr_in_ring_type)
    if is_conjugate == True:
        edge_attr_is_conjugate_type = F.one_hot(
           torch.tensor(is_conjugate_idx),
           num_classes=len(is_conjugate_dict)).to(torch.float)
        _.append(edge_attr_is_conjugate_type)


    edge_attr = torch.cat(_, dim=-1)

    edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    return edge_index, edge_attr