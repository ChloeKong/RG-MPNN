
from rdkit import Chem
path = '/home/kongyue/private-kongyue/paper-RGNN/molecular_network/transform'
import sys
sys.path.append(path)
from reduce_graph import reduceGraph
import torch
import numpy as np
from ..util import onehot


def get_pool_pair(rg_dict):

    idx = 0
    pool_index = []
    rg_name = []
    pool_pairs = []
    for key,values in rg_dict.items():
        if values == []:
            continue
        for value in values:
            for i in range(len(value)):
                pool_index.append((idx,value[i]))
            pool_pairs.append((idx, value))
            rg_name.append((key,idx))
            idx += 1
    return rg_name, pool_index, pool_pairs


def get_single_edge_index(edge_pairs,pool_pairs):

    single_edge_index = []
    for edge_pair in edge_pairs:
        edge_pair = edge_pair.cpu().numpy().tolist()
        count = 0
        for idx, pool_idx in pool_pairs:
            if set(pool_idx) >= set(edge_pair):
                count += 1
        if count == 0:
            single_edge_index.append(edge_pair)
    return single_edge_index

def get_rg_single_edge_index(single_edge_index, pool_index_no_ring):

    rg_single_edge_index = []
    for s,t in single_edge_index:
        rg_single_edge_index.append([pool_index_no_ring[s], pool_index_no_ring[t]])
    return rg_single_edge_index

def get_rg_double_edge_index(pool_pairs):

    rg_double_edge_index = []
    for i in range(len(pool_pairs)):
        for j in range(i+1, len(pool_pairs)):
            if len(set(pool_pairs[i][1]) & set(pool_pairs[j][1])) == 2:
                rg_double_edge_index.append([pool_pairs[i][0],pool_pairs[j][0]])
    return rg_double_edge_index


def get_rg_double_edge_index_attr(rg_double_edge_index):

    rg_double_edge_index_attr = []
    for [x,y] in rg_double_edge_index:
        rg_double_edge_index_attr.append([x,y,1])
        rg_double_edge_index_attr.append([y,x,1])
    return rg_double_edge_index_attr

def rg_feature(m,edge_index):

    rg_dict = reduceGraph(m)

    rg_name, pool_index, pool_pairs = get_pool_pair(rg_dict)

    pool_index_no_ring = dict((y,x) for (x,y) in pool_index)

    edge_pairs = torch.Tensor([(edge_index[0,i],edge_index[1,i]) for i in range(edge_index.shape[1])])

    single_edge_index = get_single_edge_index(edge_pairs,pool_pairs)

    rg_single_edge_index = get_rg_single_edge_index(single_edge_index,pool_index_no_ring)

    rg_double_edge_index = get_rg_double_edge_index(pool_pairs)

    rg_single_edge_index_attr = [x + [0] for x in rg_single_edge_index] 

    rg_double_edge_index_attr = get_rg_double_edge_index_attr(rg_double_edge_index)

    rg_edge_index_attr = rg_single_edge_index_attr + rg_double_edge_index_attr

    return rg_name, torch.Tensor(pool_index), torch.Tensor(rg_edge_index_attr)


edge_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE
    ]


def reduce_graph_to_mol(rg_name, rg_edge_index_attr):

    m_new = Chem.RWMol()

    # atoms
    for (a,idx) in rg_name:
        m_new.AddAtom(Chem.Atom(a))

    #bonds
    num_bonds = rg_edge_index_attr.shape[0]

    for i in range(num_bonds):
        begin_idx = int(rg_edge_index_attr[i,0])
        end_idx = int(rg_edge_index_attr[i,1])
        bond_type = edge_types[int(rg_edge_index_attr[i,2])] 
        try:
            m_new.AddBond(begin_idx, end_idx, bond_type)
        except:
            continue  

    return m_new 

rg_types = ['Mn','Y', 'Nb',
           'Fe','Zr','Mo',
           'Cr','Re','Cu',
           'Ti','Ta','Co',
           'V','W','Ni',
           'Sc','Hf','Zn']
           
def rg_x_feature(rg_name):
    rg_feats = []
    for rg,idx in rg_name:
        symbol = onehot(rg, rg_types)
        rg_feats.append(symbol)
    return torch.Tensor(rg_feats)
    
