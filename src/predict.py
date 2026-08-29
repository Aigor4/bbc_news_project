import torch
import pickle as pkl
from model import BBCModel
from etl import transform

with open("../models/vocab.pkl", "rb") as f:
    data = pkl.load(f)
vocab = data["vocab"]
max_len = data["max_len"]
names = data["names"]

model = BBCModel(vocab_size=len(vocab))
model.load_state_dict(torch.load("../models/model.pt"))
model.eval()
def predict(text):
    text = transform(text)
    words = text.split()
    indicies = [vocab.get(word, vocab["<UNK>"]) for word in words]
    if len(indicies)>max_len:
        indicies = indicies[:max_len]
    else:
        pad_len = max_len - len(indicies)
        indicies = indicies + [0] * pad_len
    tensor = torch.tensor(indicies, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        idx = torch.argmax(output, dim=1)
        return names[idx.item()]
if __name__ == "__main__":
    text = input("Enter a sentence: ")
    print(predict(text))
