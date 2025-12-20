from tokenizer import tokenize
from tf import tf
from idf import compute_idf
from tf_idf import compute_tf_idf
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

tf_idf_values = compute_tf_idf(token_frequency1,idf)

print("\n")
print("TF IDF Values : ",tf_idf_values)

