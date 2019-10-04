import hyperhyper as hy
import time
import numpy as np
import logging
import sys
import random

random.seed()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

while True:
#     sam = np.random.uniform(1e-6, 1e-4) 
    sam = 7e-5
    #window = random.choice(np.arange(8, 12))
    window = 10
    print(window)
    decay = random.choice(np.arange(0.33, 0.37, 0.01))
#     delete = random.choice([True, False])
    delete = True
    for year in [2010, 2012, 2014, 2016, 2018]:
        try:
            bunch = hy.Bunch(f'/mnt/data2/ptf/bunches/bi_{year}_decay_4')
        except Exception as e:
            try:
                print(e)
                c = hy.Corpus.from_file(f'/mnt/data2/ptf/groups/zo_bi_{year}.txt', lang='de')
                bunch = hy.Bunch(f'/mnt/data2/ptf/bunches/bi_{year}_decay_4', c)
            except:
                bunch = hy.Bunch(f'/mnt/data2/ptf/bunches/bi_{year}_decay_4')
        
        for neg in np.arange(0.5, 2, 0.1):
            print(neg, window, sam)
            for eig in np.arange(0.0, 0.6, 0.05):
                try:
                    _, res = bunch.svd(impl='scipy', pair_args={'subsample': 'deter', 'subsample_factor': sam, 'delete_oov': delete, 'decay_rate': decay, 'window': window, 'dynamic_window': 'decay'}, neg=neg, eig=eig, dim=500)
                    print(res)
                except Exception as e:
                    print('error', str(e))
                    time.sleep(10)
