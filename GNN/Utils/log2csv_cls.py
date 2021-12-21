import pandas as pd
import sys


i_path = sys.argv[1]
o_path = i_path[:-4] + '_tabel.csv'

f = open(i_path,'r')
lines = [line.strip() for line in f]

idx = 0
df = pd.DataFrame(columns=['model', 'dataset', 
        'te_AUC', 'te_MCC', 'te_ACC', 'te_SE', 'te_SP',
        'tra_AUC','tra_MCC', 'tra_ACC','tra_SE','tra_SP',
        'val_AUC','val_MCC', 'val_ACC','val_SE','val_SP',
        'batch', 'heads', 'epochs', 'x_latent_dim', 'num_passing', 'cuda', 'splits',
        'pretrain',  'checkpoints', 'path', 'data_filename', 
        'atom_types', 'atom_types_num', 'bond_types_num'])
       
for line in lines:
    if '/' in line:
        continue

    if line.startswith('模型参数'):
        data = eval(line[7:])

    if line.startswith('数据集参数'):
        data.update(eval(line[8:]))
        data['atom_types'] = str(data['atom_types'])

    if line.startswith('final training'):
        if not line.endswith('}'):
            idx = line.index('}')
            line = line[:idx+1]
        metrics = eval(line[28:])
        metrics['tra_AUC'] = metrics['AUC']
        metrics['tra_MCC'] = metrics['MCC']
        metrics['tra_ACC'] = metrics['ACC']
        metrics['tra_SE'] = metrics['SE']
        metrics['tra_SP'] = metrics['SP']
        del metrics['AUC'],metrics['MCC'],metrics['ACC'],metrics['SE'],metrics['SP']
        data.update(metrics)

    if line.startswith('final valid'):     
        if not line.endswith('}'):
            idx = line.index('}')
            line = line[:idx+1]
        metrics = eval(line[25:])
        metrics['val_AUC'] = metrics['AUC']
        metrics['val_MCC'] = metrics['MCC']
        metrics['val_ACC'] = metrics['ACC']
        metrics['val_SE'] = metrics['SE']
        metrics['val_SP'] = metrics['SP']
        del metrics['AUC'],metrics['MCC'],metrics['ACC'],metrics['SE'],metrics['SP']
        data.update(metrics)

    if line.startswith('final test'):
        if not line.endswith('}'):
            idx = line.index('}')
            line = line[:idx+1]
        metrics = eval(line[24:])
        metrics['te_AUC'] = metrics['AUC']
        metrics['te_MCC'] = metrics['MCC']
        metrics['te_ACC'] = metrics['ACC']
        metrics['te_SE'] = metrics['SE']
        metrics['te_SP'] = metrics['SP']
        del metrics['AUC'],metrics['MCC'],metrics['ACC'],metrics['SE'],metrics['SP']
        data.update(metrics)
        df = df.append(data,ignore_index=True)

df.to_csv(o_path)