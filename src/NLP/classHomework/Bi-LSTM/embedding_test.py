from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format("testDataset/word_vector/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5")
print(word_vectors["你"].shape)