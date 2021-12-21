#!/usr/bin/env python
# coding: utf-8


import numpy as np

import torch

import rdkit
from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit import rdBase
# from rdkit.Chem.rdchem import HybridizationType
# from rdkit import RDConfig
# from rdkit.Chem import ChemicalFeatures
# from rdkit.Chem.rdchem import BondType as BT
# rdBase.DisableLog('rdApp.error')




def atom_pos(suppl, i, mol):
    r'''
    '''
    text = suppl.GetItemText(i)
    N = mol.GetNumAtoms()

    pos = text.split('\n')[4:4 + N]
    pos = [[float(x) for x in line.split()[:3]] for line in pos]
    pos = torch.tensor(pos, dtype=torch.float)

    return pos

