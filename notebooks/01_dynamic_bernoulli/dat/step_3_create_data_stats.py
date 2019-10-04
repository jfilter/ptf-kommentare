import glob
import os
import numpy as np
import pickle


# Change this to the name of the folder where your dataset is
dataset_name = 'arxiv_ML'

# Change this to a list of the time slices 
time_slices = range(7,16)
time_slices = [1, 2, 3, 4, 5]

#Change this to the number of characters in the file names that should be matched to the timeslice prefix.
# i.e. if you use time_slices = [91, 92, 98, ...] 
#         use prefix_length = 2
# if you use time_slices = [1998, 1999, 2000, 2001]
#         use prefix_length = 4
prefix_length = 2

# Change this to a list of query words you would like the algorithm to print descriptive statistics of (i.e. a trajectory of the learned dynamic embeddings)
query_words = ['deep', 'inference', 'sparse', 'neuron', 'variational']
query_words = ['gewalt', 'deutschland', 'wien']

# No need to modify any code below
#######################################################
dat_stats={}
dat_stats['name'] = dataset_name
dat_stats['T_bins'] = time_slices
dat_stats['prefix'] = prefix_length
dat_stats['query_words'] = query_words
T = len(dat_stats['T_bins'])

def count_words(split):
    dat_stats[split] = np.zeros(T)
    files = glob.glob(dataset_name + '/'+ split + '/*.npy')
    for t, i  in enumerate(dat_stats['T_bins']):
        dat_files = [f for f in files if int(os.path.basename(f)[:dat_stats['prefix']]) == t]
        for fname in dat_files:
            dat = np.load(fname)
            dat_stats[split][t] += len(dat)

count_words('train')
count_words('test')
count_words('valid')

pickle.dump(dat_stats, open(dataset_name + '/dat_stats.pkl', "a+" ) )
