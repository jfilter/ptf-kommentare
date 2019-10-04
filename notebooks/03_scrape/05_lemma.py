#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import Parallel, delayed
from math import sqrt
from pathlib import Path
from lxml import etree
from tqdm import tqdm
import pickle
import dateparser

from bs4 import BeautifulSoup

import pandas as pd
import swifter

import pandas as pd
import sqlite3
from cleantext import clean

from pathlib import Path
import numpy as np
import swifter
from somajo import Tokenizer, SentenceSplitter
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

from german_lemmatizer import lemmatize

num_cores = multiprocessing.cpu_count()

import numpy as np


# In[2]:


df = pd.read_pickle('/mnt/data2/ptf/cleaned.pkl')


# In[3]:


df = df.drop(columns=['cleaned'])


# In[ ]:


slem_txt = list(lemmatize(df['text'].values[0:], n_jobs=3, chunk_size=10000))


# In[ ]:


pickle.dump( slem_txt, open( "lemma.pkl", "wb" ) )


# In[ ]:


def get_sents(texts):
    tokenizer = Tokenizer(split_camel_case=True, token_classes=False, extra_info=False)
    sentence_splitter = SentenceSplitter(is_tuple=False)
    
    results = []
    for text in texts:
#         text = clean(text, lang='de', lower=False)
        tokens = tokenizer.tokenize_paragraph(text)
        sentences = sentence_splitter.split(tokens)
        cleaned = [clean(' '.join(s), no_urls=True, no_digits=True, no_punct=True, no_line_breaks=True, lang='de') for s in sentences]
        results.append(cleaned)
    return results


# In[ ]:


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# In[ ]:


results = Parallel(n_jobs=4)(delayed(get_sents)(row) for row in tqdm(list(chunks(slem_txt, 1000))))


# In[ ]:


# cleaned = pd.read_pickle('cleaned.pkl')


# In[ ]:


# cleaned = cleaned[['cleaned']]


# In[ ]:


# cleaned = cleaned['cleaned'].values


# In[ ]:


with open('/mnt/data2/ptf/zo2.txt', 'w') as outfile:
    for d in results:
        outfile.write('\n'.join(d))


# In[ ]:




