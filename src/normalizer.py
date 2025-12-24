def compute_min_max(stats):
    features = [
        "num_tokens",
        "vocab_size",
        "avg_word_length",
        "lexical_diversity"
    ]

    min_max = {}

    for feature in features:
        values = [doc_stats[feature] for doc_stats in stats]
        min_max[feature] = (min(values), max(values))

    return min_max


def normalize_stats(stats, min_max):
    normalized = {}

    for feature, (min_value, max_value) in min_max.items():
        if max_value == min_value:
            normalized[feature] = 0.0
        else:
            normalized[feature] = (
                stats[feature] - min_value
            ) / (max_value - min_value)

    return normalized
