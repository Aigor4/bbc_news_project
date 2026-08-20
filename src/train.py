from model import BBCModel, BBCDataset, vocab, X_train, y_train, max_len
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle as pkl

train_dataset = BBCDataset(X_train, y_train, vocab, max_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = BBCModel(vocab_size=len(vocab))
cross_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = cross_loss(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "../models/model.pt")
data_to_save = {"vocab": vocab, "max_len": max_len}
with open('../models/vocab.pkl', 'wb') as f:
    pkl.dump(data_to_save, f)
