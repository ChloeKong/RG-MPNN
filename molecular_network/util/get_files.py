import os 

def get_csv_files(folder):

    g = os.walk(folder)
    filename_list = []
    for path,dir_list,file_list in g:  
        for file_name in file_list:  
            filename = os.path.join(path, file_name)
            if filename[-4:] == '.csv':
                filename_list.append(filename)
    print('共含有%i个csv文件'%len(filename_list))
    return filename_list

def get_pkl_files(folder):

    g = os.walk(folder)
    filename_list = []
    for path,dir_list,file_list in g:  
        for file_name in file_list:  
            filename = os.path.join(path, file_name)
            if filename[-4:] == '.pkl' or 'pkl.gz':
                filename_list.append(filename)
    print('%i pkl or pkl.gz files in total'%len(filename_list))
    return filename_list