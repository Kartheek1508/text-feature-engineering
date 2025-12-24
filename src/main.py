from tokenizer import tokenize
from tf import tf
from idf import compute_idf
from tf_idf import compute_tf_idf
from stats import compute_stats
from vectorizer import vectorize
from similarity import cosine_similarity
from normalizer import compute_min_max, normalize_stats
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


tfidf_vec1 = vectorize(tf_idf_value1, vocab)
tfidf_vec2 = vectorize(tf_idf_value2, vocab)
tfidf_vec3 = vectorize(tf_idf_value3, vocab)

print("\n")
print("Cosine similarity(Doc1 VS Doc2) : ",cosine_similarity(tfidf_vec1,tfidf_vec2))
print("Cosine similarity(Doc2 VS Doc3) : ",cosine_similarity(tfidf_vec2,tfidf_vec3))
print("Cosine similarity(Doc1 VS Doc3) : ",cosine_similarity(tfidf_vec1,tfidf_vec3))


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

#ranking the documents for a qurey

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


#


#making stats vector

stats_vec1 = [stats_doc1["num_tokens"],stats_doc1["vocab_size"],stats_doc1["avg_word_length"],stats_doc1["lexical_diversity"]]
stats_vec2 = [stats_doc2["num_tokens"],stats_doc2["vocab_size"],stats_doc2["avg_word_length"],stats_doc2["lexical_diversity"]]
stats_vec3 = [stats_doc3["num_tokens"],stats_doc3["vocab_size"],stats_doc3["avg_word_length"],stats_doc3["lexical_diversity"]]
#unified_vec1 = vectorize([tf_idf_value1,tf_idf_value2,tf_idf_value3,stats_doc1["num_tokens"],stats_doc1["vocab_size"],stats_doc1["avg_word_length"],stats_doc1["lexical_diversity"]],vocab)
#unified_vec2 = [tf_idf_value1,tf_idf_value2,tf_idf_value3,stats_doc2["num_tokens"],stats_doc2["vocab_size"],stats_doc2["avg_word_length"],stats_doc2["lexical_diversity"]]
#unified_vec3 = [tf_idf_value1,tf_idf_value2,tf_idf_value3,stats_doc3["num_tokens"],stats_doc3["vocab_size"],stats_doc3["avg_word_length"],stats_doc3["lexical_diversity"]]
#print(unified_vec1)
unified_vec1 = tfidf_vec1 + stats_vec1
unified_vec2 = tfidf_vec2 + stats_vec2
unified_vec3 = tfidf_vec3 + stats_vec3

query_stats = compute_stats(query_tokens)
query_stats_vec = [query_stats["num_tokens"],query_stats["vocab_size"],query_stats["avg_word_length"],query_stats["lexical_diversity"]]

unified_qurey_vec = query_vector + query_stats_vec

unified_scores = [cosine_similarity(unified_qurey_vec,unified_vec1)
,cosine_similarity(unified_qurey_vec,unified_vec2)
,cosine_similarity(unified_qurey_vec,unified_vec3)
]

#sorted_unified_scores = sorted(unified_scores,reverse=True)

print("\n")
print("ranking files based on qurey")

doc = 0
for unified_score in unified_scores:
        print(f"doc{doc+1} -> {unified_score}")
        doc = doc+1


all_stats = [stats_doc1,stats_doc2,stats_doc3]

min_max = compute_min_max(all_stats)

norm_stats_doc1 = normalize_stats(stats_doc1,min_max)
norm_stats_doc2 = normalize_stats(stats_doc2,min_max)
norm_stats_doc3 = normalize_stats(stats_doc3,min_max)

norm_qurey_stats = normalize_stats(query_stats,min_max)

print("norm_stats_doc1 keys:", norm_stats_doc1.keys())
print("norm_stats_doc1 full:", norm_stats_doc1)


stats_vec_norm1 = [norm_stats_doc1["num_tokens"],norm_stats_doc1["vocab_size"],norm_stats_doc1["avg_word_length"],norm_stats_doc1["lexical_diversity"]]

unified_norm_vec1 = tfidf_vec1+stats_vec_norm1
print("Normalized doc1 stats:", norm_stats_doc1)

unified_norm_vec2 = tfidf_vec2+[
    norm_stats_doc2["num_tokens"],
    norm_stats_doc2["vocab_size"],
    norm_stats_doc2["avg_word_length"],
    norm_stats_doc2["lexical_diversity"]
]
unified_norm_vec3 = tfidf_vec3+[
    norm_stats_doc3["num_tokens"],
    norm_stats_doc3["vocab_size"],
    norm_stats_doc3["avg_word_length"],
    norm_stats_doc3["lexical_diversity"]
]
norm_query_stats_vec = [
    norm_qurey_stats["num_tokens"],
    norm_qurey_stats["vocab_size"],
    norm_qurey_stats["avg_word_length"],
    norm_qurey_stats["lexical_diversity"]
]

unified_norm_query_vec = query_vector + norm_query_stats_vec


unified_norm_scores = [cosine_similarity(unified_norm_query_vec,unified_norm_vec1)
,cosine_similarity(unified_norm_query_vec,unified_norm_vec2)
,cosine_similarity(unified_norm_query_vec,unified_norm_vec3)
]

print("\n")
print("ranking files based on qurey(normalized stats)")

doc = 0
for unified_score in unified_norm_scores:
        print(f"doc{doc+1} -> {unified_score}")
        doc = doc+1