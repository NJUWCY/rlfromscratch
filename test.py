import numpy as np 
import h5py 

with h5py.File('/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/minari_dataset/converted/ant/expert-v0.hdf5', 'r') as f:

    print(sum(f['rewards'][:1000]) )
    # for key in f.keys(): 
    #     print(key, f[key].shape) 
    #     if isinstance(f[key], h5py.Dataset): 
    #         data = f[key][()]  # 读取数据集的内容 
    #         print(data)  # 打印数据集的内容