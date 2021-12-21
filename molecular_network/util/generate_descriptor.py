import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
import numpy as np
import copy
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
import argparse

def Morgan(input_path, output_path, radius= 2,nbits=1024):
    data = pd.read_csv(input_path)
    try:
        Chem.PandasTools.AddMoleculeColumnToFrame(data, smilesCol='SMILES', molCol='MOL', includeFingerprints=True)
    except:
        print('Rdkit can\'t recognize some smiles')
        pass
    descriptors_list = []
    moldata = copy.deepcopy(data)
    for i in range(len(data)):
        info = {}
        mol = data['MOL'][i]
        morgan_fps = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits, bitInfo = info,useFeatures = False)#useFeatures = True为FCFP4
        descriptors_list.append(morgan_fps.ToBitString())
    Morgen = np.zeros(shape=(len(data),nbits),dtype=int)
    for m in range(len(descriptors_list)):
        for n in range(len(descriptors_list[m])):
            morgan_fp = descriptors_list[m]
            if morgan_fp[n] == '1':
                Morgen[m][n] = 1
    column_name = range(0,nbits)
    FP_DF = pd.DataFrame(Morgen,columns = column_name)
    DF_end = pd.concat([moldata,FP_DF],axis=1)
    DF_end.to_csv(output_path,index=False)
    print('Finish Morgan fingerpint!')


def MACCS(input_path,output_path):
    data = pd.read_csv(input_path)
    moldata = copy.deepcopy(data)
    try:
        Chem.PandasTools.AddMoleculeColumnToFrame(data, smilesCol='SMILES', molCol='MOL', includeFingerprints=True)
    except:
        print('Rdkit can\'t recognize some smiles')
        pass
    descriptors_list = []    
    for i in range(len(data)):
        mol = data['MOL'][i]
        fps = Chem.MACCSkeys.GenMACCSKeys(mol)
        descriptors_list.append(fps.ToBitString())
    fp_array = np.zeros(shape=(len(data),len(descriptors_list[0])),dtype=int)
    for m in range(len(descriptors_list)):
        for n in range(len(descriptors_list[m])):
            fp_ = descriptors_list[m]
            if fp_[n] == '1':
                fp_array[m][n] = 1
    column_name = range(0,len(descriptors_list[0]))
    FP_DF = pd.DataFrame(fp_array,columns = column_name)
    DF_end = pd.concat([moldata,FP_DF],axis=1)
    DF_end.to_csv(output_path,index=False)
    print('Finish MACCS!')

def RDKFingerprint(input_path,output_path):
    data = pd.read_csv(input_path)
    moldata = copy.deepcopy(data)
    try:
        Chem.PandasTools.AddMoleculeColumnToFrame(data, smilesCol='SMILES', molCol='MOL', includeFingerprints=True)
    except:
        print('Rdkit can\'t recognize some smiles')
        pass
    descriptors_list = []    
    for i in range(len(data)):
        mol = data['MOL'][i]
        fps = Chem.RDKFingerprint(mol)
        descriptors_list.append(fps.ToBitString())
    fp_array = np.zeros(shape=(len(data),len(descriptors_list[0])),dtype=int)
    for m in range(len(descriptors_list)):
        for n in range(len(descriptors_list[m])):
            fp_ = descriptors_list[m]
            if fp_[n] == '1':
                fp_array[m][n] = 1
    column_name = range(0,len(descriptors_list[0]))
    FP_DF = pd.DataFrame(fp_array,columns = column_name)
    DF_end = pd.concat([moldata,FP_DF],axis=1)
    DF_end.to_csv(output_path,index=False)
    print('Finish RDKFingerprint!')

def All_molecular_property(input_path,output_path):
    data = pd.read_csv(input_path)
    moldata = copy.deepcopy(data)
    try:
        Chem.PandasTools.AddMoleculeColumnToFrame(data, smilesCol='SMILES', molCol='MOL', includeFingerprints=True)
    except:
        print('Rdkit can\'t recognize some smiles')
        pass
    property_name = ['MolLogP','MolMR','MolWt','HeavyAtomCount','NHOHCount','NOCount','NumHAcceptors','NumHDonors',
                     'NumHeteroatoms','NumRotatableBonds','NumValenceElectrons','RingCount',
                     'FractionCSP3','TPSA','LabuteASA']
    fp_array = np.zeros(shape=(len(data),len(property_name)))  
    for i in range(len(data)):
        mol = data['MOL'][i]
        descriptors_list = []
        property_ = Descriptors.MolLogP(mol)
        descriptors_list.append(property_)

        property_ = Descriptors.MolMR(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.MolWt(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.HeavyAtomCount(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.NHOHCount(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.NOCount(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.NumHAcceptors(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.NumHDonors(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.NumHeteroatoms(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.NumRotatableBonds(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.NumValenceElectrons(mol)
        descriptors_list.append(property_)
           
        property_ = Descriptors.RingCount(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.FractionCSP3(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.TPSA(mol)
        descriptors_list.append(property_)
        
        property_ = Descriptors.LabuteASA(mol)
        descriptors_list.append(property_)

        fp_array[:][i] = descriptors_list
    FP_DF = pd.DataFrame(fp_array,columns = property_name)
    DF_end = pd.concat([moldata,FP_DF],axis=1)
    DF_end.to_csv(output_path,index=False)
    print('Finish molecular property descriptors!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate descriptors or fingerprints')
    parser.add_argument('-i',help='path of input csv file with "SMILES" column')
    parser.add_argument('-o',help='path of output csv file')
    parser.add_argument('-des',default='MACCS',choices=['MACCS','ECFP','rdkit','property'],help='MACCS, ECFP, rdkit or property')
    args = parser.parse_args()

    input_csv = args.i
    out_csv = args.o
    des = args.des

    if des == 'MACCS':
        print('calculating MACCS fingerprints')
        MACCS(input_csv,out_csv)

    elif des == 'ECFP':
        print('calculating ECFP fingerprints')
        Morgan(input_csv,out_csv)

    elif des == 'rdkit':
        print('calculating rdkit descriptors')
        RDKFingerprint(input_csv,out_csv)

    elif des == 'property':
        print('calculating rdkit molecular property descriptors')
        All_molecular_property(input_csv,out_csv)

    else:
        print('only can calculate MACCS, ECFP, rdkit or property')

