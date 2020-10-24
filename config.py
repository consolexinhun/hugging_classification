import torch
import os, sys
# 设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 类别转索引和索引转类别
CLASS_2_IDX = {'财经':0, '房产':1, '家居':2, '教育':3, '科技':4, '时尚':5, '时政':6}
IDX_2_CLASS = {0:'财经', 1:'房产', 2:'家居', 3:'教育', 4:'科技', 5:'时尚', 6:'时政'}

# 具体是几分类

NUM_CLASSES = len(CLASS_2_IDX)

# 预训练模型名字
MODEL_NAME = "hfl/chinese-xlnet-base"
# 句子中最多多少个单词
MAXLEN = 192
# 保存模型的路径
OUTPUT_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "output_model"))

# 训练参数配置
BATCH_SIZE = 2
EPOCH = 2
# 训练测试文件
TRAIN_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "labeled_data.csv"))
TEST_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "test_data.csv"))





