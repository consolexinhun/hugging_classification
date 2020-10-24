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
TRAIN_FILE, MAXLEN, TEST_FILE


class CustomDataset(Dataset):
    def __init__(self, data, maxlen, with_labels, model_name):
        '''
        :param data: 数据集
        :param maxlen: 句子最大长度
        :param with_labels: 是否携带标签
        :param model_name: 模型名字
        '''
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = str(self.data.loc[index, "content"])

        tokenizer = self.tokenizer(sentence,
                                   padding="max_length",  # 填充到最大长度
                                   truncation=True,       # 超过最大长度就截断
                                   max_length=self.maxlen,
                                   return_tensors="pt")

        token_ids = tokenizer["input_ids"].squeeze(0)                   # token的id
        attention_mask = tokenizer["attention_mask"].squeeze(0)          # 如果填充就是0，不填充就是1
        token_type_ids = tokenizer["token_type_ids"].squeeze(0)         # 如果是单句都为1，如果是句对，前0后1

        if self.with_labels:
            label = self.data.loc[index, "class_label"]
            return token_ids, attention_mask, token_type_ids, label
        else:
            return token_ids, attention_mask, token_type_ids

def process_data(filename, class_2_idx, with_labels=True):
    data = pd.read_csv(filename, encoding="utf-8")
    if with_labels:
        data = data.replace({"class_label": class_2_idx})
    return data


def main():
    all_data = process_data(filename=TRAIN_FILE, class_2_idx=CLASS_2_IDX, with_labels=True)
    print(all_data.shape)


all_data = process_data(filename=TRAIN_FILE, class_2_idx=CLASS_2_IDX, with_labels=True)
data = CustomDataset(all_data, maxlen=MAXLEN, with_labels=True, model_name=MODEL_NAME)

train_df, val_df = train_test_split(data, test_size=0.2, shuffle=True, random_state=1)
# train_data = CustomDataset(train_df, maxlen=MAXLEN,with_labels=True, model_name=MODEL_NAME)
train_loader = DataLoader(train_df, batch_size=BATCH_SIZE, shuffle=True)
# val_data = CustomDataset(val_df, maxlen=MAXLEN,with_labels=True, model_name=MODEL_NAME)
val_loader = DataLoader(val_df, batch_size=BATCH_SIZE)


test_df = process_data(filename=TEST_FILE, class_2_idx=CLASS_2_IDX, with_labels=False)
test_data = CustomDataset(test_df, maxlen=MAXLEN, with_labels=False, model_name=MODEL_NAME)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
