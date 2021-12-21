from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import copy
import difflib
import itertools
import os
# fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
fdefName = '/home/kongyue/private-kongyue/RGNN-develop/RG_BaseFeatures.fdef'
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
import pandas as pd
import numpy as np
def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx()))
    return mol

def show_baseFeature(fdefName):
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    print(len(factory.GetFeatureFamilies()))
    print(len(factory.GetFeatureDefs()))
    print(factory.GetFeatureFamilies())
# show_baseFeature(fdefName)
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def show_fra(factory):
    family_df = pd.DataFrame(columns=['family', 'definition']) 
    family_names = factory.GetFeatureFamilies()
    dict_fra = factory.GetFeatureDefs()
    for k,v in dict_fra.items():
        for fam in family_names:
            if fam in k:
                family_df.loc[k] = [fam, v]
    return family_df

def get_pharm(m):
    atom_num = m.GetNumAtoms()
    feats = factory.GetFeaturesForMol(m)
    dict_feats = {}
    for f in feats:
        dict_feats[f.GetFamily()] = dict_feats.get(f.GetFamily(),[])+[list(f.GetAtomIds())]  


    return dict_feats, atom_num


def similar(s1, s2): 
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def rm_dupli(l):
    l_new = []
    for item in l:
        if item not in l_new:
            l_new.append(item)
    return l_new
    

def single_merge_list(l):
    if len(l)==0:
        return l
    for j in range(1,len(l)):
        if set(l[0]) & set(l[j]) != set():
            l[0] = list(set(l[0])|set(l[j]))
            del l[j]
            break
    else:
        new_l.append(l[0])
        del l[0]

    return l

def x_merge(feat_dict, group_name):

    if group_name not in feat_dict.keys():
        return feat_dict
    
    l = feat_dict[group_name]  
    l = rm_dupli(l)
    global new_l  
    new_l = []
    while True:
        l = single_merge_list(l)
        if len(l) == 0:
            break
    feat_dict[group_name] = new_l
    return feat_dict


def y_merge(feat_dict, high_name, low_name):

    if (high_name not in feat_dict.keys()) or (low_name not in feat_dict.keys()):
        return feat_dict

    high = feat_dict[high_name]
    low = feat_dict[low_name]

    for i,x in enumerate(high):
        for j,y in enumerate(low):
            if (set(x)&set(y)) != set():
                high[i] = list(set(x)|set(y))
                low[j] = []       
    return feat_dict

def rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, high_name, low_name, rg_name):

    if (high_name not in feat_dict.keys()) or (low_name not in feat_dict.keys()):
        return rg_dict, feat_hit_dict

    high = feat_dict[high_name]
    low = feat_dict[low_name]

    for x in high:
        for y in low:
            if (set(x)&set(y)) != set():
                feat_hit_dict[high_name].append(x)
                feat_hit_dict[low_name].append(y)

                rg_dict[rg_name].append(list(set(x)|set(y))) 
            
    return rg_dict, feat_hit_dict

def AD_merge(rg_dict, D_name, A_name, rg_name):

    if (D_name not in rg_dict.keys()) or (A_name not in rg_dict.keys()):
        return rg_dict
    D = rg_dict[D_name]
    A = rg_dict[A_name]
    for i,x in enumerate(D):
        for j,y in enumerate(A):
            if (set(x)&set(y)) != set():
                rg_dict[rg_name].append(list(set(x)|set(y)))
                D[i] = []
                A[j] = []
                
    return rg_dict

def non_hit_define(feat_dict, feat_hit_dict, rg_dict, feat_name, rg_name):

    if feat_name not in feat_dict.keys():
        return rg_dict

    rg_dict[rg_name].extend([item for item in feat_dict[feat_name] if item not in feat_hit_dict[feat_name]])

    return rg_dict


def reduceGraph(m):

    rg_dict ={
           'Mn':[],'Y':[],'Nb':[],
           'Fe':[],'Zr':[],'Mo':[],
           'Cr':[],'Re':[],'Cu':[],
           'Ti':[],'Ta':[],'Co':[],
           'V':[],'W':[],'Ni':[],
           'Sc':[],'Hf':[],'Zn':[]
           }

    rg_dict_new ={
           'Mn':[],'Y':[],'Nb':[],
           'Fe':[],'Zr':[],'Mo':[],
           'Cr':[],'Re':[],'Cu':[],
           'Ti':[],'Ta':[],'Co':[],
           'V':[],'W':[],'Ni':[],
           'Sc':[],'Hf':[],'Zn':[]
           }
           
    feat_dict, atom_num = get_pharm(m)
    feat_hit_dict = dict([(key, []) for key in list(feat_dict.keys())])  

    feat_dict = x_merge(feat_dict, 'CC')
    feat_dict = x_merge(feat_dict, 'Acceptor')
    feat_dict = x_merge(feat_dict, 'NegIonizable')
    feat_dict = x_merge(feat_dict, 'PosIonizable')
    feat_dict = x_merge(feat_dict, 'Donor')

    feat_dict = y_merge(feat_dict,'PosIonizable','NegIonizable')
    feat_dict = y_merge(feat_dict,'PosIonizable','Donor')
    feat_dict = y_merge(feat_dict,'PosIonizable','Acceptor')
    feat_dict = y_merge(feat_dict,'NegIonizable','Donor')
    feat_dict = y_merge(feat_dict,'NegIonizable','Acceptor')

    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aromatic', 'PosIonizable', 'Mn')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aromatic', 'NegIonizable', 'Fe')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aromatic', 'Donor', 'Ti')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aromatic', 'Acceptor', 'V')

    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aliphatic', 'PosIonizable', 'Y')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aliphatic', 'NegIonizable', 'Zr')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aliphatic', 'Donor', 'Ta')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aliphatic', 'Acceptor', 'W')

    rg_dict = y_merge(rg_dict,'Mn','Fe')
    rg_dict = y_merge(rg_dict,'Mn','Ti')
    rg_dict = y_merge(rg_dict,'Mn','V')
    rg_dict = y_merge(rg_dict,'Fe','Ti')
    rg_dict = y_merge(rg_dict,'Fe','V')

    rg_dict = y_merge(rg_dict,'Y','Zr')
    rg_dict = y_merge(rg_dict,'Y','Ta')
    rg_dict = y_merge(rg_dict,'Y','W')
    rg_dict = y_merge(rg_dict,'Zr','Ta')
    rg_dict = y_merge(rg_dict,'Zr','W')

    for x in ['Mn','Fe','Ti','V','Y','Zr','Ta','W']:
        rg_dict = x_merge(rg_dict,x)

    rg_dict = AD_merge(rg_dict, 'Ti', 'V', 'Cr') 
    rg_dict = AD_merge(rg_dict, 'Ta', 'W', 'Re') 

    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'Aromatic', 'Sc')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'Aliphatic', 'Hf')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'PosIonizable', 'Nb')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'NegIonizable', 'Mo')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'Donor', 'Co')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'Acceptor', 'Ni')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'CC', 'Zn')

    rg_dict = AD_merge(rg_dict, 'Co', 'Ni', 'Cu') 

    rg_dict = x_merge(rg_dict, 'Zn')

    hit_num = []
    for x in rg_dict.values():
        if x != []:
            for y in x:
                hit_num.extend(y)

    C_num = [x for x in list(range(atom_num)) if x not in list(set(hit_num))]
    for i in C_num:
        rg_dict['Zn'].append([i])

    for key,values in rg_dict.items():
        if values == []:
            continue
        for value in values:
            if value == []:
                continue
            if value not in rg_dict_new[key]:
                rg_dict_new[key].append(value)

    return rg_dict_new
