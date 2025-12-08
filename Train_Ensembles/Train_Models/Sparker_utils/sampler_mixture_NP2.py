import numpy as np
import time, datetime

# NP2 signal generator                                                                                              
def NP2_gen(size, seed=None, random_gen=None):
    if size>10000:
        raise Warning('Sample size is grater than 1000: Generator will not approximate the tale well')
    sample = np.array([])
    if seed==None:
        seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
    if random_gen ==None:
        random_gen = np.random.default_rng(seed=seed)
    #normalization factor                                                                                         
    #np.random.seed(seed)                                                                                            
    Norm = 256.*0.25*0.25*np.exp(-2)
    while(len(sample)<size):
        x = random_gen.uniform(0,1) #assuming not to generate more than 10 000 events          
                                                                                                                     
        p = random_gen.uniform(0, Norm)
        if p<= 256.*x*x*np.exp(-8.*x):
            sample = np.append(sample, x)
    return sample
