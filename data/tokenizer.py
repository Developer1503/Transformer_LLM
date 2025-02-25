class MultiModalTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def text_to_sequence(self, text):
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in text.split()]

    def encode(self, text, max_length, padding="post"):
        sequence = self.text_to_sequence(text)
        if len(sequence) < max_length:
            if padding == "post":
                sequence.extend([self.vocab["<pad>"]] * (max_length - len(sequence)))
            else:
                sequence = [self.vocab["<pad>"]] * (max_length - len(sequence)) + sequence
        else:
            sequence = sequence[:max_length]
        return sequence
