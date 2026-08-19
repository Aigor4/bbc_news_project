from etl import load
from collections import Counter
import torch

X_train, X_test, y_train, y_test = load("../data/bbc_data.csv")
max_len = int(X_train.str.split().str.len().quantile(0.95))
splited = X_train.str.split()

word_list = []
for review in splited:
    for word in review:
        word_list.append(word)

word_counts = Counter(word_list)
min_freq = 2
filtered = []
for word, count in word_counts.most_common():
    if count >= min_freq:
        filtered.append((word, count))

vocab = {"<PAD>": 0, "<UNK>": 1}
for i, (word, count) in enumerate(filtered, start=2):
    vocab[word] = i

class BBCDataset(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train, vocab, max_len):
        self.X_train = X_train
        self.y_train = y_train
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.X_train)
    def __getitem__(self, index):
        text = self.X_train.iloc[index]
        words = text.split()
        indices = []
        for word in words:
            indices.append(self.vocab.get(word, self.vocab["<UNK>"]))

        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            pad_len = self.max_len - len(indices)
            indices = indices + [0] * pad_len

        label = self.y_train.iloc[index]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)