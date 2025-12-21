def vectorize(tfidf_dict, vocab):
    vector = []
    for word in vocab:
        vector.append(tfidf_dict.get(word, 0.0))
    return vector
