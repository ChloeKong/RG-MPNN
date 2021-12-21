import pandas as pd
import sys


i_path = sys.argv[1]
o_path = i_path[:-4] + '_tabel.csv'

f = open(i_path,'r')
lines = [line.strip() for line in f]

idx = 0
df = pd.DataFrame(columns=['model', 'dataset', 
        'te_MSE', 'te_RMSE', 'te_MAE', 'te_r2',
        'tra_MSE', 'tra_RMSE', 'tra_MAE', 'tra_r2',
        'val_MSE', 'val_RMSE', 'val_MAE', 'val_r2',
        'batch', 'heads', 'epochs', 'x_latent_dim', 
        'num_passing', 'num_passing_before','num_passing_after',
        'num_passing_pool','cuda', 'splits','pretrain',  
        'checkpoints', 'path', 'data_filename', 
        'atom_types', 'atom_types_num', 'bond_types_num'])
       
for line in lines:
    if line.startswith('模型参数'):
        data = eval(line[7:])

    if line.startswith('数据集参数'):
        data.update(eval(line[8:]))
        data['atom_types'] = str(data['atom_types'])

    if line.startswith('final training'):
        metrics = eval(line[28:])
        metrics['tra_MSE'] = metrics['MSE']
        metrics['tra_RMSE'] = metrics['RMSE']
        metrics['tra_MAE'] = metrics['MAE']
        metrics['tra_r2'] = metrics['r2']
        del metrics['MSE'],metrics['RMSE'],metrics['MAE'],metrics['r2']
        data.update(metrics)

    if line.startswith('final valid'):
        metrics = eval(line[25:])
        metrics['val_MSE'] = metrics['MSE']
        metrics['val_RMSE'] = metrics['RMSE']
        metrics['val_MAE'] = metrics['MAE']
        metrics['val_r2'] = metrics['r2']
        del metrics['MSE'],metrics['RMSE'],metrics['MAE'],metrics['r2']
        data.update(metrics)

    if line.startswith('final test'):
        metrics = eval(line[24:])
        metrics['te_MSE'] = metrics['MSE']
        metrics['te_RMSE'] = metrics['RMSE']
        metrics['te_MAE'] = metrics['MAE']
        metrics['te_r2'] = metrics['r2']
        del metrics['MSE'],metrics['RMSE'],metrics['MAE'],metrics['r2']
        data.update(metrics)
        df = df.append(data,ignore_index=True)

df.to_csv(o_path)