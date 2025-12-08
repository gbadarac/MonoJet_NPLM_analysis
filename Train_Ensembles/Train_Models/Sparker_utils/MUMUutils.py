import numpy as np
import h5py
import os

def BuildSample_DY(INPUT_PATH, seed=-1, features=[], N_Events=-1, nfiles=20, shuffle=True):
    if seed>0:
        np.random.seed(seed)
    #random integer to select Zprime file between n files                                          
    u = np.arange(nfiles)#np.random.randint(100, size=100)                                         
    if shuffle: np.random.shuffle(u)
    toy_label = INPUT_PATH.split("/")[-2]
    print(toy_label)
    HLF = np.array([])
    for u_i in u:
        if not os.path.exists(INPUT_PATH+toy_label+str(u_i+1)+".h5"): continue
        f    = h5py.File(INPUT_PATH+toy_label+str(u_i+1)+".h5", 'r')
        keys = list(f.keys())
        if u_i==u[0]:
            print('available features: ', keys)
        if len(keys)==0: continue #check whether the file is empty                                 
        cols = np.array([])
        if len(features): keys = features
        for i in range(len(keys)):
            feature = np.array(f.get(keys[i]))
            feature = np.expand_dims(feature, axis=1)
            if i==0: cols = feature
            else: cols = np.concatenate((cols, feature), axis=1)
        if shuffle: np.random.shuffle(cols) #don't want to select always the same event first 
        if HLF.shape[0]==0:
            HLF=cols
            i+=1
        else: HLF=np.concatenate((HLF, cols), axis=0)
        f.close()
        if N_Events>0 and HLF.shape[0]>=N_Events:
            HLF=HLF[:N_Events, :]
            break
    print(HLF.shape)
    #return HLF[:, [4, 5, 1, 2, 0, 3]]                                                             
    return HLF