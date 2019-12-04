import hyperhyper as hy
import time
import numpy as np
import logging
import sys
import random

random.seed()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

while True:
    sam = np.random.uniform(1e-6, 1e-4) 
    window = 10
    #window = 2
    #window = random.choice(np.arange(2, 11, 1))
    print(window)
    decay = 0.35
    decay = random.choice(np.arange(0.2, 0.4, 0.005))
#     delete = random.choice([True, False])
    delete = True
    for year in [2010]:
        try:
            bunch = hy.Bunch(f'/mnt/data/bunches/all2')
        except Exception as e:
            try:
                print(e)
                c = hy.Corpus.from_text_files(f'/mnt/data/groups/all/', lang='de', keep_n=100000, preproc_func=hy.tokenize_texts)
                bunch = hy.Bunch(f'/mnt/data/bunches/all2', c)
            except Exception as e:
                print('error', e)
                bunch = hy.Bunch(f'/mnt/data/bunches/all2')
        
        for neg in [0.5, 1, 1.5, 2, 3, 4, 5]:
            print(neg, window, sam)
            for eig in np.arange(0.0, 0.6, 0.05):
                try:
                    _, res = bunch.svd(impl='scipy', pair_args={'subsample': 'deter', 'subsample_factor': sam, 'delete_oov': delete, 'window': window, 'decay_rate': decay, 'dynamic_window': 'decay'}, low_memory=True, neg=neg, eig=eig, dim=500)
                    print(res)
                except Exception as e:
                    print('error', str(e))
                    time.sleep(10)
