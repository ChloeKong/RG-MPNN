#!/usr/bin/env python
#!/usr/bin/env python
# coding: utf-8


from ..util import onehot,ringsize
import numpy as np

import torch
# import torch.nn.functional as F
# from torch_sparse import coalesce

#from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
# import rdkit
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit import rdBase
# from rdkit.Chem.rdchem import HybridizationType
# from rdkit import RDConfig
# from rdkit.Chem import ChemicalFeatures
# from rdkit.Chem.rdchem import BondType as BT
rdBase.DisableLog('rdApp.error')



atom_types = ['H', 'C', 'N', 'O','F', 'S', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Fe', 'Se']
hybrid_types = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
exp_val_types = [0,1,2,3,4,5,6,7]
ring_size_types = [3,4,5,6,7,8]
Hs_types = [0,1,2,3,4,5]


def atom_feature(mol,
                 atom_types=atom_types,
                 hybrid=True,
                 ring_size=False,
                 aromatic=True,
                 mass=False,
                 atom_num=False,
                 exp_val=True,
                 Hs=True
                 ):

    atoms = mol.GetAtoms()
    ri = mol.GetRingInfo()
    ri_a = ri.AtomRings()
    x = []

    for atom in atoms:
        
        feats = []

        if isinstance(atom_types,int): 
            _ = list(range(atom_types))
            symbol = onehot(atom.GetAtomicNum()-1, _)
            feats.append(symbol)
        else:
            symbol = onehot(atom.GetSymbol(), atom_types)  
            feats.append(symbol)

        if hybrid == True:
            _ = onehot(str(atom.GetHybridization()),hybrid_types,other=True)  
            feats.append(_)
        
        if aromatic == True:
            _ = [atom.GetIsAromatic()+0]  
            feats.append(_)

            
        if exp_val == True:
            _ = onehot(atom.GetExplicitValence(),exp_val_types,other=True) 
            feats.append(_) 
        
        if Hs == True:
            _ = onehot(atom.GetTotalNumHs(includeNeighbors=True),Hs_types,other=True) 
            feats.append(_)
        
        if ring_size == True:
            _ = ringsize(atom, ri_a)  
            feats.append(_)
            
            
        if atom_num == True:
            _ = [(atom.GetAtomicNum())/35]  
            feats.append(_) 

        if mass == True:
            _ = [(atom.GetMass())/80]  
            feats.append(_) 

        x_ = np.concatenate(feats)

        x.append(x_)

    return torch.Tensor(x)
