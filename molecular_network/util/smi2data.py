
from rdkit import Chem
import sys

sys.path.append('/home/tuotuoyue/private-kongyue/MN')
sys.path.append('/home/hyper2020/MN')
from molecular_network.mol_feature import atom_feature, bond_feature
from torch_geometric.data import Data

def smi2data(smiles):
   
    smis = []
    data_list = []
    canonical_smis = []
    if type(smiles) == str:
        smis.append(smiles)
    if type(smiles) == list:
        smis = smiles

    for smi in smis:

        mol = Chem.MolFromSmiles(smi)
        canonical_smi = Chem.MolToSmiles(mol)
        canonical_smis.append(canonical_smi)

        mol = Chem.MolFromSmiles(canonical_smi)
        mol = Chem.AddHs(mol)
        
        x = atom_feature(mol)

        edge_index, edge_attr = bond_feature(mol) 

        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr)

        data_list.append(data)
    return canonical_smis,data_list