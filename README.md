# Word-Embeddings

Take the Sinhala/Tamil corpus (according to the language you prefer) given in Homework 1. Using Gensim (https://radimrehurek.com/gensim/models/word2vec.html) create both skipgram and CBOW word vectors. If model training takes too long, consider using a GPU, or reducing the data set size.

Using these two types of word vectors, find similar words for a set of 10  (common) words that you choose from the corpus. Compare the accuracy of the models.

Now use the Pre-trained FastText vectors (https://github.com/facebookresearch/fastText/blob/master/pre-trained-vectors.md) for Sinhala and Tamil. For the set of words you select above, try finding similar words. Try both bin+txt and txt versions. Record your findings.

Submission:

***A report  containing 

1. How gensim was used to create word vectors.

2. How similarity between words was measured

3. Your analysis on the results

*** Code for

1. Word vector creation using Gensim

2. Similarity measurement between words
