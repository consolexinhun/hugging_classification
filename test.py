import os, sys, csv
import random

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel

from config import DEVICE,CLASS_2_IDX, IDX_2_CLASS,MODEL_NAME, OUTPUT_MODEL,BATCH_SIZE, \
TRAIN_FILE, MAXLEN, NUM_CLASSES, EPOCH, DEVICE
from dataprocess import test_loader
from model import MyModel, AdamW

def test(model, test_loader, with_label):
    ckpt = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), OUTPUT_MODEL, "best.pt")))
    model.load_state_dict(ckpt["model_state_dict"])
    print("-------------TEST----------------")
    predicts = []

    model.eval()
    for batch in test_loader:
        token_ids, attention_mask, token_type_ids = tuple(p.to(DEVICE) for p in batch)
        with torch.no_grad():
            logits = model(token_ids, attention_mask, token_type_ids)
            logit = logits.detach().cpu().numpy()

            preds = np.argmax(logit, axis=-1).flatten()
            predicts.extend(preds)

    return predicts

def to_csv(predicts):
    rel_dict = {'财经': '高风险', '时政': '高风险',
                '房产': '中风险', '科技': '中风险',
                '教育': '低风险', '时尚': '低风险', '游戏': '低风险',
                '家居': '可公开','体育': '可公开', '娱乐': '可公开'}

    ids = list(range(len(predicts)))
    class_labels = [IDX_2_CLASS[p] for p in predicts]
    rank_labels = [rel_dict[c] for c in class_labels]

    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","class_label","rank_label"])
        writer.writerows(zip(ids, class_labels, rank_labels))

model = MyModel(isFreeze=False, model_name=MODEL_NAME, hidden_size=768, num_classes=NUM_CLASSES).to(DEVICE)
predicts = test(model, test_loader, with_label=False)
to_csv(predicts)