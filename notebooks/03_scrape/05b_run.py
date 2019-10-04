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

sents = pickle.load( open( "s.pkl", "rb" ) )

slem_txt = list(lemmatize(sents, n_jobs=4, chunk_size=50000))

pickle.dump( slem_txt, open( "s_l.pkl", "wb" ) )
