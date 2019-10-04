### Dynamic Bernoulli Embeddings for Language Evolution

This repository contains scripts for running (dynamic) Bernoulli embeddings on text data.
They have been run and tested on Linux. 

To execute, go into the source folder (`src/`) and run 

   ```python main.py --dynamic True --fpath [path/to/data]```

substitute the path to the folder where you put the data for `[path/to/data]`.
The data folder and files have to be structured in a specific format.
For your convenience, we included some scripts that will help you preprocess the text data in `dat/src/`. For instructions on the required data format see `dat/README.md`.

For all commandline options run:

   ```python main.py --help```

For fastest convergence we recommend the following 2-step training procedure.
First run

   ```python main.py --fpath [path/to/data]```

This executes Bernoulli embeddings without dynamics. The scripts uses the current timestamp to create a folder where the results are saved ([path/to/results/]). We will use these results to initialize the dynamic embeddings:

   ```python main.py --dynamic True --fpath [path/to/data] --init [path/to/result]/alpha_constant```

Make sure to use the same `--K` for both runs.


### Reference
Maja Rudolph and David Blei, 2017. [Dynamic Bernoulli Embeddings for Language Evolution](https://arxiv.org/abs/1703.08052).
arxiv preprint arxiv:1703.08052.
