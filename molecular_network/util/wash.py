from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
import sys
import argparse


def _InitialiseNeutralisationReactions():
    patts= (
        # Imidazoles
        ('[n+;H]','n'),
        # Amines
        ('[N+;!H0]','N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]','O'),
        # Thiols
        ('[S-;X1]','S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]','N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]','N'),
        # Tetrazoles
        ('[n-]','[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]','S'),
        # Amides
        ('[$([N-]C=O)]','N'),
        )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

def NeutraliseCharges(mol, reactions=None, replaced=True):
    if reactions is None:
        _reactions=_InitialiseNeutralisationReactions()
        reactions=_reactions
        
    for i,(reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol,True))
    else:
        return mol
    
def wash(cpd_smi_list):

    smis = []
    valid_idx = []
    remover = SaltRemover()
    for i,smi in enumerate(cpd_smi_list):
        try:
            m = Chem.MolFromSmiles(smi)
            assert m is not None
            m = remover.StripMol(m,dontRemoveEverything=True)
            m = NeutraliseCharges(m)
            smis.append(Chem.MolToSmiles(m,canonical=True))
            valid_idx.append(i)
        except:
            #smis.append('invalid')
            print('Invalid smiles(index={}): {}'.format(i,smi))
    print("Valid: {}".format(len(valid_idx)))
    print("Invalid: {}".format(len(cpd_smi_list)-len(valid_idx)))
    return smis,valid_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='1. remove salts 2. neutralize 3. generate canonical smiles')
    parser.add_argument('input',help='input csv file with smiles column')
    parser.add_argument('-smi_name',default='smiles',help='name of smiles column')
    parser.add_argument('-label_name',default='label',help='name of label column')
    args = parser.parse_args()
    
    smi = args.smi_name
    label = args.label_name
    path = args.input
    
    df = pd.read_csv(path)
    smis = df[smi].tolist()
    smis, valid_idx = wash(smis)
    df_valid = df.iloc[valid_idx,:]
    df_new = df_valid.copy() 
    df_new[smi] = smis
    df_new.rename(columns={smi:'smiles',label:'label'},inplace=True) 
    df_new[['smiles','label']].to_csv(path[:-4]+'_washed.csv', sep=",", header=True, index=False)
    
    
    
    
    
