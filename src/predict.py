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