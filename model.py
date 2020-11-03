import torch
from torch import nn

from transformers import AutoModel

class MyModel(nn.Module):
    def __init__(self, isFreeze, model_name, hidden_size, num_classes):
        '''
        :param isFreeze: 是否冻结预训练模型的参数
        :param model_name: 模型名字
        :param hidden_size: 隐藏层
        :param num_classes: 几分类
        '''
        super(MyModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name,output_hidden_states=True,return_dict=True)
        if isFreeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*4, num_classes)
        )


    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        ##  hidden_states:是一个tuple，每一层返回：(batch_size, sequence_length, hidden_size) 只有在output_hidden_states=True时才有返回
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1) # 将最后四层cat，在hiddensize上拼接
        last_four = hidden_states[:, 0, :]  # 只需要取第一个单词的输出
        logits = self.fc(last_four)
        return logits

        # return outputs.logits