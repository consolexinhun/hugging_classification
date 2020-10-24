import os, sys
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
from dataprocess import train_loader, val_loader
from model import MyModel, AdamW

best_acc = 0


def train_eval(model, criteon, optimizer, train_loader, val_loader, epochs):
    print("-----------------start traing-------------------")

    for epoch in range(epochs):
        model.train()
        print("Epoch: %d" % epoch)

        for i, batch in enumerate(train_loader):
            token_ids, attention_mask, token_type_ids, labels = tuple(t.to(DEVICE) for t in batch)
            logits = model(token_ids, attention_mask, token_type_ids)

            loss = criteon(logits, labels)
            print("*******  i:%s  **********" % loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                eval(model, optimizer, val_loader)

def eval(model, optimizer, val_loader):
    model.eval()

    all_item, correct_item = 0, 0

    for batch in val_loader:
        token_ids, attention_mask, token_type_ids, labels = tuple(t.to(DEVICE) for t in batch)
        with torch.no_grad():
            logits = model(token_ids, attention_mask, token_type_ids)
            logit = logits.detach().cpu().numpy()    # [batch, num_classes]
            label_id = labels.cpu().numpy()           # [batch ]

            all_item += len(logits)
            correct_item += np.sum(np.argmax(logits, axis=-1).flatten() == label_id.flatten())
    acc = correct_item/all_item

    print("$$$$$$$$$$   Val-Acc: %s  $$$$$$$$$$$" % acc)
    global best_acc
    if acc > best_acc:
        best_acc = acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, OUTPUT_MODEL)

        print("the best model save in %s" % OUTPUT_MODEL)


model = MyModel(isFreeze=False, model_name=MODEL_NAME, hidden_size=768, num_classes=NUM_CLASSES)

criteon = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

train_eval(model, criteon, optimizer, train_loader, val_loader, EPOCH)




