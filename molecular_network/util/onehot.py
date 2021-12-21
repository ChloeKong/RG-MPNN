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
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
rdBase.DisableLog('rdApp.error')




def onehot(c, cat, other=False):  
    length = len(cat)
    cat = dict(zip(cat,list(range(length))))
    if other == False:
        onehot = np.zeros(length)
        if c in cat:
            onehot[cat[c]] = 1
        assert onehot.sum() == 1,
    if other == True:
        onehot = np.zeros(length+1)
        if c in cat:
            onehot[cat[c]] = 1
        else:
            onehot[-1] = 1
        assert onehot.sum() == 1 
    return onehot



def ringSize_a(a, rings, max_size = 8, min_size = 3):  
    onehot = np.zeros(max_size - min_size + 1)
    aid = a.GetIdx()
    for ring in rings: 
        if aid in ring and len(ring) <= max_size:
        
            onehot[len(ring) - min_size] += 1
    return onehot