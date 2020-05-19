# PTF-Kommentare

This repository contains code and notes for my [Prototype Fund](https://prototypefund.de/project/hasskommentare-automatisiert-filtern/) project. It was mainly done between 01.03.2019 and 01.09.2019. The topic: Explaining machine learning and natural language processing at the example of news comments, and visualize language change.

## Sub Projects

The work is devided into serveral sub projects:

- Website for explaining ML and NLP, as well as investigating language change in online comments: [kommentare.vis.one](https://kommentare.vis.one), [code](https://github.com/jfilter/kommentare.vis.one)
- Backend to serve local views on word embeddings (used for kommentare.vis.one): [ptf-kommentare-backend](https://github.com/jfilter/ptf-kommentare-backend)
- Python package to construct (stable) word embeddings for small data: [hyperhyper](https://github.com/jfilter/hyperhyper)
- Python package to clean text: [clean-text](https://github.com/jfilter/clean-text)
- Python package for common text preprocessing for German: [german](https://github.com/jfilter/german-preprocessing)
- Python package to lemmatize German text: [german-lemmatizer](https://github.com/jfilter/german-lemmatizer)
- Benchmark for SVD implementations: [sparse-svd-benchmark](https://github.com/jfilter/sparse-svd-benchmark)

## Create your own Visualizations of Language Change

Here is quick guide on how to create your own videos.

1. Divide your data in time slices & create a word embedding for each slice
2. Save the embedding in `KeyedVectors` format of [gensim](https://radimrehurek.com/gensim/) (using [hyperhyper](https://github.com/jfiler/hyperhyper) to create stable word embeddings is advised)
3. Install [ffmpeg](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg)
4. `pip install git+https://github.com/jfilter/adjustText && pip install gensim scikit-learn matplotlib colormath`
5. Adopt the code in this notebook (so you also need to have either [Jupyter Lab](https://jupyter.org/) or [Jupyter Notebook](https://jupyter.org/) installed.)

Right now, it's not _that_ easy to create those videos. However, it's doable and I'm willing to help you, if you want to use my code. The 'important' part of the code is commented thoroughly. Please [contact me](mailto:hi@jfilter.de) for assistance.

Two Papers for more scientific background:

- [Hamilton et al.: Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change](https://www.aclweb.org/anthology/P16-1141/)
- [Hellrich et al.: The Influence of Down-Sampling Strategieson SVD Word Embedding Stability](https://www.aclweb.org/anthology/W19-2003/)

Some more papers [here](https://github.com/jfilter/ptf-kommentare/blob/master/notes/references.md).

## Sponsoring

This work was funded by the German [Federal Ministry of Education and Research](https://www.bmbf.de/en/index.html).

<img src="./bmbf_funded.svg">
