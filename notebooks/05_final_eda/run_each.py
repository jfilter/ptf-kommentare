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
    #window = random.choice(np.arange(5, 11))
    window=11
    print(window)
    decay = 0.35
    #decay = random.choice(np.arange(0.2, 0.4, 0.005))
#     delete = random.choice([True, False])
    delete = True
    for year in range(2010, 2016):
        try:
            bunch = hy.Bunch(f'/mnt/data2/ptf/bunches/new_{year}')
        except Exception as e:
            try:
                print(e)
                c = hy.Corpus.from_file(f'/mnt/data2/ptf/groups/zo_{year}_cleaned.txt', lang='de')
                bunch = hy.Bunch(f'/mnt/data2/ptf/bunches/new_{year}', c)
            except:
                bunch = hy.Bunch(f'/mnt/data2/ptf/bunches/new_{year}')
        
        for neg in [0.5, 1, 1.5, 2]:
            print(neg, window, sam)
            for eig in np.arange(0.0, 0.6, 0.05):
                try:
                    _, res = bunch.svd(threads=2, impl='scipy', pair_args={'subsample': 'deter', 'subsample_factor': sam, 'delete_oov': delete, 'decay_rate': decay, 'window': window, 'dynamic_window': 'decay'}, neg=neg, eig=eig, dim=500)
                    print(res)
                except Exception as e:
                    print('error', str(e))
                    time.sleep(10)
