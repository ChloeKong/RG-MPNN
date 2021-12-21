#!/usr/bin/env python
# coding: utf-8


import rdkit
from rdkit import Chem
import numpy as np

def ringSize_a(a, rings, max_size = 8, min_size = 3):  
    onehot = np.zeros(max_size - min_size + 1)
    aid = a.GetIdx()
    for ring in rings:
        if aid in ring and len(ring) <= max_size:

            onehot[len(ring) - min_size] += 1
    return onehot

def ringsize(atom, rings):  
    bits = np.zeros(7)
    id = atom.GetIdx()
    for ring in rings:  
        a = len(ring)
        if a > 8:
            a = 9
        if id in ring:
            bits[a - 3] += 1
    return bits