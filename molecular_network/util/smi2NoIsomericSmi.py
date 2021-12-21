import pandas as pd 
from rdkit import Chem

def smarts2smiles(input):
    'input: smiles or smarts'
    'e.g. input [O-:1][N+:2](=[O:3])[c:4]1[c:5]([OH:6])[cH:7][cH:8][c:9]([Cl:10])[cH:11]1'
    'output: O=[N+]([O-])c1cc(Cl)ccc1'
    try:
        if ':1]' in input: 
            # print('smarts found :', input)
            m = Chem.MolFromSmarts(input)
            for a in m.GetAtoms(): a.SetAtomMapNum(0) 
        else:
            m = Chem.MolFromSmiles(input)
        
        smi = Chem.MolToSmiles(m,isomericSmiles=False) 

    except:
        smi = False 
        print('wrong molecule: ', input)
    return smi


def smi2NoIsomericSmi(df):

    df.columns = ['smiles','label']
    smi = df.smiles
    smis = []
    for s in smi:
        m = Chem.MolFromSmiles(s)
        s_new = Chem.MolToSmiles(m,isomericSmiles=False)
        smis.append(s_new)
    df.smiles = smis
    return df


