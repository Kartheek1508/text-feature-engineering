from tokenizer import tokenize
from tf import tf
from idf import compute_idf
from tf_idf import compute_tf_idf
from stats import compute_stats
from vectorizer import vectorize
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH1 = os.path.join(BASE_DIR, 'data', 'doc1.txt')
DATA_PATH2 = os.path.join(BASE_DIR, 'data', 'doc2.txt')
DATA_PATH3 = os.path.join(BASE_DIR, 'data', 'doc3.txt')

with open(DATA_PATH1,'r') as f:
    data1 = f.read()

#data 1
tokens1 = tokenize(data1)
print("Tokens : \n",tokens1)

token_frequency1 = tf(tokens1)
print("TF : \n",token_frequency1)

#data2

with open(DATA_PATH2,'r') as f:
    data2 = f.read()

tokens2 = tokenize(data2)
print("Tokens : \n",tokens2)

token_frequency2 = tf(tokens2)
print("TF : \n",token_frequency2)

#data3

with open(DATA_PATH3,'r') as f:
    data3 = f.read()
    
tokens3 = tokenize(data3)
print("Tokens : \n",tokens3)

token_frequency3 = tf(tokens3)
print("TF : \n",token_frequency3)

token_list = [tokens1, tokens2, tokens3]
idf = compute_idf(token_list)
print("IDF values : \n",idf)

tf_idf_value1 = compute_tf_idf(token_frequency1,idf)
print("\n")
print("TF IDF Values : ",tf_idf_value1)

tf_idf_value2 = compute_tf_idf(token_frequency2,idf)
print("TF IDF Values : ",tf_idf_value2)

tf_idf_value3 = compute_tf_idf(token_frequency3,idf)
print("TF IDF Values : ",tf_idf_value3)

stats_doc1 = compute_stats(tokens1)
stats_doc2 = compute_stats(tokens2)
stats_doc3 = compute_stats(tokens3)

print(stats_doc1)

def build_vocab(tfidf_dicts):
    vocab = set()
    for tfidf in tfidf_dicts:
        vocab.update(tfidf.keys())
    return sorted(vocab)

vocab = build_vocab([tf_idf_value1,tf_idf_value2,tf_idf_value3])


v1 = vectorize(tf_idf_value1, vocab)
v2 = vectorize(tf_idf_value2, vocab)
v3 = vectorize(tf_idf_value3, vocab)

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
print("\n")
print("Cosine similarity(Doc1 VS Doc2) : ",cosine_similarity(v1,v2))
print("Cosine similarity(Doc2 VS Doc3) : ",cosine_similarity(v2,v3))
print("Cosine similarity(Doc1 VS Doc3) : ",cosine_similarity(v1,v3))


documents = {
    "doc1": tf_idf_value1,
    "doc2": tf_idf_value2,
    "doc3": tf_idf_value3
}

vocab = build_vocab(list(documents.values()))

doc_vectors = {
    name: vectorize(tfidf, vocab)
    for name, tfidf in documents.items()
}

query = "python programming language"

query_tokens = tokenize(query)
query_tf = tf(query_tokens)
query_tfidf = compute_tf_idf(query_tf, idf)
query_vector = vectorize(query_tfidf, vocab)

#ranking the documents

scores = []
values = doc_vectors
for name,vector in values.items():
    score = cosine_similarity(query_vector,vector)
    scores.append((name,score))

scores = sorted(scores, key=lambda x: x[1], reverse=True)

print("Query:", query)
print("\nRanked results:")
for name,score in scores:
    print(f"{name} → {score:.4f}")
