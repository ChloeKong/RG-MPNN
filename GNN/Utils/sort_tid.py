import numpy as np
from collections import Counter

def sort_tid(tid):
    '''
    把不连续的tid转成连续的需序号
    '''
    t_unique_list = torch.unique(tid)
    num = t_unique_list.size()[0]
    t_num = torch.arange(num)
    t_dict = ()
    for i in tid:


