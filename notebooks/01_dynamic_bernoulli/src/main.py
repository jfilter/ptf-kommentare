from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
import tensorflow as tf
import time

from data import *
from models import *
from args import *
from utils import *


args = parse_args()

d = bern_emb_data(args.cs, args.ns, args.fpath, args.dynamic, args.n_epochs)

dir_name = make_dir(d.name)


m = define_model(args, d, dir_name)

m.initialize_training()

m.train_embeddings()
