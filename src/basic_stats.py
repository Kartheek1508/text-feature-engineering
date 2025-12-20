import os

with open(r'/Users/bnskartheek/Programming/text_feature_eng/text-feature-engineering/data/test.txt', 'r') as f:
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
