import numpy as np
import os
import pandas as pd
import pickle
import glob

class bern_emb_data():
    def __init__(self, cs, ns, fpath, dynamic, n_epochs=1, remove_stopwords=False):
        assert cs%2 == 0
        self.cs = cs
        self.ns = ns
        self.n_epochs = n_epochs
        self.dynamic = dynamic
        dat_stats = pickle.load(open(os.path.join(fpath, "dat_stats.pkl"), "a+"))
        self.T = len(dat_stats['T_bins'])
        self.name = dat_stats['name']
        if not self.dynamic:
            self.N = np.sum(dat_stats['train']).astype('int32')
            self.n_train = (self.N/n_epochs).astype('int32')
            self.n_valid = np.sum(dat_stats['valid']).astype('int32')
            self.n_test = np.sum(dat_stats['test']).astype('int32')
        else:
            self.N = np.sum(dat_stats['train']).astype('int32')
            self.n_train = np.maximum(dat_stats['train']/n_epochs, 2*np.ones_like(dat_stats['train'])).astype('int32')
            self.n_valid = dat_stats['valid'].astype('int32')
            self.n_test = dat_stats['test'].astype('int32')

	# load vocabulary
        df = pd.read_csv(os.path.join(fpath, 'unigram.txt'), delimiter='\t',header=None)
        self.labels = df[0].values
        self.counts = df[len(df.columns)-1].values
        counts = (1.0 * self.counts / self.N) ** (3.0 / 4)
        self.unigram = counts / self.N
        self.w_idx = range(len(self.labels))
        if remove_stopwords:
            sw_df = pd.read_csv(os.path.join(fpath, 'stop_words.txt'), delimiter='\t',header=None)
            stop_words = sw_df[0].values 
            self.w_idx = [i for i, w in enumerate(self.labels) if w not in stop_words]
            self.labels = self.labels[self.w_idx]
            self.counts = self.counts[self.w_idx]
            self.unigram = self.unigram[self.w_idx]
            self.unigram_t = np.load(os.path.join(fpath,'unigram_t.npy'))[:,self.w_idx]
            self.unigram_t = self.unigram_t/self.unigram_t.sum(axis=0)
        self.L = len(self.labels)
        self.dictionary = dict(zip(self.labels,range(self.L)))
        self.query_words = [w for w in dat_stats['query_words'] if w in self.labels]

        # data generator (training)
        train_files = glob.glob(os.path.join(fpath,'train','*.npy'))
        if self.dynamic:
            self.batch = [0]*self.T
            for t, i in enumerate(dat_stats['T_bins']):
                print(t, i)
                self.batch[t] = self.batch_generator(self.n_train[t] + self.cs, [f for f in train_files if int(os.path.basename(f)[:dat_stats['prefix']]) == i])
        else:
           self.batch = self.batch_generator(self.n_train + self.cs, train_files)

        # data generator (validation)
        valid_files = glob.glob(os.path.join(fpath,'valid','*.npy'))
        if self.dynamic:
            self.valid_batch = [0]*self.T
            for t, i in enumerate(dat_stats['T_bins']):
                self.valid_batch[t] = self.batch_generator(self.n_valid[t] + self.cs, [f for f in valid_files if int(os.path.basename(f)[:dat_stats['prefix']]) == i])
        else:
           self.valid_batch = self.batch_generator(self.n_valid + self.cs, valid_files)

        # data generator (test)
        test_files = glob.glob(os.path.join(fpath,'test','*.npy'))
        if self.dynamic:
            self.test_batch = [0]*self.T
            for t, i in enumerate(dat_stats['T_bins']):
                self.test_batch[t] = self.batch_generator(self.n_test[t] + self.cs, [f for f in test_files if int(os.path.basename(f)[:dat_stats['prefix']]) == i])
        else:
           self.test_batch = self.batch_generator(self.n_test + self.cs, test_files)

    def load_file(self, fn):
        with open(fn, 'r') as myfile:
            words = myfile.read().replace('\n', '').split()
        data = np.zeros(len(words))
        for idx, word in enumerate(words):
            if word in self.dictionary:
                data[idx] = self.dictionary[word]
        return data

    def batch_generator(self, batch_size, files):
        f_idx = 0
        #data = self.load_file(files[f_idx])
        data = np.load(files[f_idx])
        while True:
            if data.shape[0] < batch_size:
                f_idx+=1
                if (f_idx>=len(files)):
                    f_idx = 0
        	#data_new = self.load_file(files[f_idx])
                data_new = np.load(files[f_idx])
                data = np.hstack([data, data_new])
                if data.shape[0] < batch_size:
                    continue
            words = data[:batch_size]
            data = data[batch_size:]
            yield words
    
    def train_feed(self, placeholder):
        if self.dynamic:
            feed_dict = {}
            for t in range(self.T):
                feed_dict[placeholder[t]] = self.batch[t].next()
            return feed_dict
        else:
            return {placeholder: self.batch.next()}

    def valid_feed(self, placeholder):
        if self.dynamic:
            feed_dict = {}
            for t in range(self.T):
                feed_dict[placeholder[t]] = self.valid_batch[t].next()
            return feed_dict
        else:
            return {placeholder: self.valid_batch.next()}

    def test_feed(self, placeholder):
        if self.dynamic:
            feed_dict = {}
            for t in range(self.T):
                feed_dict[placeholder[t]] = self.test_batch[t].next()
            return feed_dict
        else:
            return {placeholder: self.test_batch.next()}

