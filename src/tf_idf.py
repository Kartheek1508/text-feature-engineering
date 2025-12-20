def compute_tf_idf(type_f,idf):
    tf_idf = {}

    #print(type_f)
    for words in type_f:
        tf_idf[words]=type_f[words]*idf[words]

    return tf_idf