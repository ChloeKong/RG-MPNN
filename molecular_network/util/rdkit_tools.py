from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def mols2grid_image(mols, molsPerRow, legends=None):   
    'mols转image' 
    mols = [e if e is not None else Chem.RWMol() for e in mols]    
    for mol in mols:
        AllChem.Compute2DCoords(mol)
    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(200, 200), legends=legends)
