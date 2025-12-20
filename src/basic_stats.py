import os
from collections import Counter

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'test.txt')

with open(DATA_PATH, 'r') as f:
    data = f.read()
    print("Length of data:", len(data))
    print("Raw data repr:", repr(data))

lines = 1
for i in range(len(data)):
    if(data[i] == '\n'):
        lines = lines+1

print("Number of lines:", lines)

words = 1
for i in range(len(data)):
    if(data[i]== " ") or (data[i]=='\n'):
        words = words + 1

print("Number of words:", words)

data_lower = data.lower()
data_split = data_lower.split()

tokens = []
for word in data_lower.split():
    cleaned = word.strip('.,!?";:()[]{}')
    if cleaned:
        tokens.append(cleaned)

print("Processed words list:", tokens)


c = Counter(tokens)
print("Word frequencies:", c)

tf = {}
for words in c:
    tf[words] = c[words]/len(tokens)

for word in tf:
    print(f"{word}: TF = {tf[word]:.4f}")

