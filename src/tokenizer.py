def tokenize(text):
    
    tokens = []
    text = text.lower()
    for word in text.split():
        cleaned = word.strip('.,!?";:(){}[]')
        tokens.append(cleaned)
    return tokens