from collections import Counter
def tf(tokens):
    count = Counter(tokens)

    tf = {}
    for words in count:
        tf[words] = count[words]/len(tokens)

    return tf