import os, sys, re
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

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

    data.content = data["content"].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5]+', "", x))
    return data

def expand():
    labels, contents = [], []
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "cnews.train.txt")), "r", encoding="utf-8") as f:
        for line in f:
            label, content = line.split("\t")
            labels.append(label)
            contents.append(content)

    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "cnews.test.txt")), "r", encoding="utf-8") as f:
        for line in f:
            label, content = line.split("\t")
            labels.append(label)
            contents.append(content)

    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "cnews.val.txt")), "r", encoding="utf-8") as f:
        for line in f:
            label, content = line.split("\t")
            labels.append(label)
            contents.append(content)

    data_dict = {
        "id": list(range(len(labels))),
        "class_label": [CLASS_2_IDX[c] for c in labels],
        "content": contents
        # "content": [re.sub(r'[^\u4e00-\u9fa5]+', "", content) for content in contents]
    }
    expand_data = pd.DataFrame(data_dict)
    expand_data = expand_data[["id", "class_label", "content"]] # 调整列的顺序，没有什么实质作用
    return expand_data


origin_data = process_data(filename=TRAIN_FILE, class_2_idx=CLASS_2_IDX, with_labels=True)
expand_data = expand()

all_data = pd.concat((origin_data, expand_data), axis=0, ignore_index=True) # 忽视之前的索引，也就是重建索引

data = CustomDataset(all_data, maxlen=MAXLEN, with_labels=True, model_name=MODEL_NAME)

train_df, val_df = train_test_split(data, test_size=0.3, shuffle=True, random_state=1)
train_loader = DataLoader(train_df, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_df, batch_size=BATCH_SIZE)


test_df = process_data(filename=TEST_FILE, class_2_idx=CLASS_2_IDX, with_labels=False)
test_data = CustomDataset(test_df, maxlen=MAXLEN, with_labels=False, model_name=MODEL_NAME)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
