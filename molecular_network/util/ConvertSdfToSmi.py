from rdkit import Chem
import sys

f_sdf = sys.argv[1]
f_smi = f_sdf[:-4] + '.smi'

suppl = Chem.SDMolSupplier(f_sdf)

n = 0
with open(f_smi,'w') as f:
    f.write('smiles\n')
    for m in suppl:
        f.write(Chem.MolToSmiles(m))
        f.write('\n')
        if n % 1 == 0:
            print('processed %i molecules' %(n))
        n += 1
