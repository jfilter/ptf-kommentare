# References

## Omer Levy et al. (Goldberg): Improving Distributional Similaritywith Lessons Learned from Word Embeddings

Word2Vec is not superior to SVD, it depends on the hyper paramters. Simple count-based approaches with SVD are fine and also work well for small corpora sizes.

https://www.aclweb.org/anthology/Q15-1016

https://theworldofdatascience.wordpress.com/word2vec-vs-svd-to-compare-words/

https://news.ycombinator.com/item?id=15502859

## Hellrich et al. (Jena)

### The Influence of Down-Sampling Strategieson SVD Word Embedding Stability, Hellrich et al.

Buid upon Omer Levy's work to improve the stability of word embeddings.

https://arxiv.org/pdf/1808.06810.pdf

Here is a modified code that implements the used method:

https://github.com/hellrich/hyperwords

Here are full experiments:

https://github.com/hellrich/embedding_downsampling_comparison

### Don’t Get Fooled by Word Embeddings—Better Watch their Neighborhood, Hellrich et al.

Word2Vec etc. are unstable.

https://dh2017.adho.org/abstracts/487/487.pdf

### Bad Company – Neighborhoods in Neural Embedding Spaces Considered Harmful

https://aclweb.org/anthology/C16-1262

### Jeseme

Some Java programm to visualize change of Words

http://aclweb.org/anthology/C18-2003

http://jeseme.org/search?word=car&corpus=coha

## histwords (Stanford)

Old buy maybe usefull visualisations, align Word-embeddings of time-sliced

https://www.aclweb.org/anthology/P16-1141

https://github.com/williamleif/histwords

## Dynamic Word Embeddings

There are several papers and implementation for dynamic word embeddings. Since there are recent and inspired by Word2Vec, it is likely that they as well suffer from the stability issues. So better leave them out.

### DynamicWord2Vec

https://github.com/yifan0sun/DynamicWord2Vec

### Diachronic word embeddings and semantic shifts: a survey

https://www.aclweb.org/anthology/C18-1117

### Dynamic Word Embeddings

No code.

https://arxiv.org/abs/1702.08359

### Dynamic Bernoulli Embeddings

http://delivery.acm.org/10.1145/3190000/3185999/p1003-rudolph.pdf?ip=95.91.244.173&id=3185999&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1556101640_117c5b78e93a905c12a5eb40fe3a65af

https://github.com/mariru/dynamic_bernoulli_embeddings

### Temporal Embeddings

http://delivery.acm.org/10.1145/3160000/3159703/p673-yao.pdf?ip=141.89.221.168&id=3159703&acc=ACTIVE%20SERVICE&key=2BA2C432AB83DA15%2E1F4E6143780147C9%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1551377073_3d4f2efa54a90cbbb440cf4d11978a36

https://arxiv.org/abs/1806.03537

Links about temporal embeddings: https://github.com/manuyavuz/temporal-embeddings

# Background

## PMI

https://en.wikipedia.org/wiki/Pointwise_mutual_information

## SVD

https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/

A fast SVD implementation: https://github.com/gbolmier/funk-svd

https://stats.stackexchange.com/questions/108156/understanding-singular-value-decomposition-in-the-context-of-lsi

https://stats.stackexchange.com/questions/4735/what-are-the-differences-among-latent-semantic-analysis-lsa-latent-semantic-i/28149

https://blog.statsbot.co/singular-value-decomposition-tutorial-52c695315254

https://www.reddit.com/r/MachineLearning/comments/9scd1m/d_svd_just_as_good_as_word2vec_where_to_find_the/

## LSI

http://stefan-kaufmann.uconn.edu/Papers/SagiKaufmannClark_MoutonLSA_revision_vE4.pdf

## Word2Vec

http://ruder.io/secret-word2vec/

## Align Matrix

https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
