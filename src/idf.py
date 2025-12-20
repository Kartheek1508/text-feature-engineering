import math
def compute_idf(tokens_list):
    total_doc = len(tokens_list)
    doc_freq = {}

    for tokens in tokens_list:
        unique_words = set(tokens)
        for word in unique_words:
            doc_freq[word] = doc_freq.get(word,0)+1

    idf_values = {}
    for word,df in doc_freq.items():
        idf_values[word] = math.log(total_doc/df)

    return idf_values