# -*- coding: utf-8 -*-
import os
import csv
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.utils import shuffle as reset
from tqdm import tqdm
import transformers
from transformers import *

transformers.logging.set_verbosity_error()
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
classes2idx = {'财经': 0, '房产': 1, '家居': 2, '教育': 3, '科技': 4, '时尚': 5, '时政': 6}
idx2classes = {0: '财经', 1: '房产', 2: '家居', 3: '教育', 4: '科技', 5: '时尚', 6: '时政'}

model_name = 'hfl/chinese-xlnet-base'
output_model = './models/model.pth'
best_score = 0
batch_size = 32


class CustomDataset(Data.Dataset):
    def __init__(self, data, maxlen, with_labels=True, model_name='bert-base-chinese'):
        self.data = data  # pandas dataframe

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = str(self.data.loc[index, 'content'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects

        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'class_label']
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


class MyModel(nn.Module):
    def __init__(self, freeze_bert=False, model_name='bert-base-chinese', hidden_size=768, num_classes=2):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 4, num_classes, bias=False),
        )

    def forward(self, input_ids, attn_masks, token_type_ids):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attn_masks)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]),
                                  dim=-1)  # [bs, seq_len, hidden_dim*4]
        first_hidden_states = hidden_states[:, 0, :]  # [bs, hidden_dim*4]
        logits = self.fc(first_hidden_states)
        return logits


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_data(filename, classes2idx, with_labels=True):
    data = pd.read_csv(filename, encoding='utf-8')
    if with_labels:
        data = data.replace({'class_label': classes2idx})
    return data


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()  # [3, 5, 8, 1, 2, ....]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)
    print('The best model has been saved')


def train_test_split(data_df, test_size=0.2, shuffle=True, random_state=None):
    if shuffle:
        data_df = reset(data_df, random_state=random_state)

    train = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    test = data_df[:int(len(data_df) * test_size)].reset_index(drop=True)

    return train, test


def train_eval(model, criterion, optimizer, train_loader, val_loader, epochs=2):
    # checkpoint = torch.load(output_model, map_location='cpu')

    # model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print('-----Training-----')
    for epoch in range(epochs):
        model.train()
        print('Epoch', epoch)
        for i, batch in enumerate(tqdm(train_loader)):
            batch = tuple(t.to(device) for t in batch)
            logits = model(batch[0], batch[1], batch[2])
            loss = criterion(logits, batch[3])
            print(i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                eval(model, optimizer, val_loader)


def eval(model, optimizer, validation_dataloader):
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], batch[1], batch[2])
            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    global best_score
    if best_score < eval_accuracy / nb_eval_steps:
        best_score = eval_accuracy / nb_eval_steps
        save(model, optimizer)


def test(model, dataloader, with_labels=False):
    # load model
    checkpoint = torch.load(output_model, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print('-----Testing-----')

    pred_label = []
    model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], batch[1], batch[2])
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            pred_label.extend(preds)

    pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv('pred.csv', header=['class_label'],
                                                                       encoding='utf-8')

    rel_dict = {'财经': '高风险', '时政': '高风险', '房产': '中风险', '科技': '中风险', '教育': '低风险', '时尚': '低风险', '游戏': '低风险', '家居': '可公开',
                '体育': '可公开', '娱乐': '可公开'}
    with open('pred.csv', encoding='utf-8') as f:
        rows = [row for row in csv.reader(f)]
        rows = np.array(rows[1:])  # all data, 3D
        label_list = [label for _, label in rows]  # label list
        final_col = []
        for i in label_list:
            final_col.append(rel_dict[idx2classes[int(i)]])

        data = pd.read_csv('pred.csv')
        data['final'] = final_col
        data = data.replace({'class_label': idx2classes})
        data.to_csv("result.csv", index=False, encoding="utf_8_sig", header=['id', 'class_label', 'rank_label'])

    print('Test Completed')


if __name__ == '__main__':
    set_seed(1)  # Set all seeds to make results reproducible

    data_df = process_data('datasets/labeled_data.csv', classes2idx, True)

    train_df, val_df = train_test_split(data_df, test_size=0.2, shuffle=True, random_state=1)

    print("Reading training data...")
    train_set = CustomDataset(train_df, maxlen=192, model_name=model_name)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, num_workers=5, shuffle=True)

    print("Reading validation data...")
    val_set = CustomDataset(val_df, maxlen=192, model_name=model_name)
    val_loader = Data.DataLoader(val_set, batch_size=batch_size, num_workers=5, shuffle=True)

    model = MyModel(freeze_bert=False, model_name=model_name, hidden_size=768, num_classes=len(classes2idx))
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    train_eval(model, criterion, optimizer, train_loader, val_loader, epochs=10)

    # print("Reading test data...")
    # test_df = process_data('test_data.csv', classes2idx, False)
    # test_set = CustomDataset(test_df, maxlen=192, with_labels=False, model_name=model_name)
    # val_loader = Data.DataLoader(test_set, batch_size=batch_size, num_workers=5, shuffle=False)
    #
    # test(model, val_loader, with_labels=False)