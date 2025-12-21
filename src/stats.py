def compute_stats(tokens):
    total_tokens = len(tokens)
    unique_tokens = set(tokens)
    vocab_size = len(unique_tokens)

    if total_tokens == 0:
        return {
            "num_tokens": 0,
            "vocab_size": 0,
            "avg_word_length": 0.0,
            "lexical_diversity": 0.0
        }


    total_chars = sum(len(word) for word in tokens)
    avg_word_length = total_chars / total_tokens


    lexical_diversity = vocab_size / total_tokens

    stats = {
        "num_tokens": total_tokens,
        "vocab_size": vocab_size,
        "avg_word_length": avg_word_length,
        "lexical_diversity": lexical_diversity
    }

    return stats
