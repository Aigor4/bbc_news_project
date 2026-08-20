from model import *
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

with open("../models/vocab.pkl", "rb") as f:
    data = pkl.load(f)
vocab = data["vocab"]
max_len = data["max_len"]
model = BBCModel(vocab_size=len(vocab))
model.load_state_dict(torch.load("../models/model.pt"))
model.eval()

test_dataset = BBCDataset(X_test, y_test, vocab, max_len)
test_loader = DataLoader(test_dataset, batch_size=64)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

average = 'macro'

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds, average=average))
print("Recall:", recall_score(all_labels, all_preds, average=average))
print("F1:", f1_score(all_labels, all_preds, average=average))
print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds))