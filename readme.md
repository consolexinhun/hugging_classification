# 项目结构

```bash
root@5f2d7693679a:/home# tree -L 2
.
|-- config.py  （配置文件）
|-- dataprocess.py  （数据预处理）
|-- datasets        （数据集）
|   |-- labeled_data.csv
|   |-- test_data.csv
|   `-- unlabeled_data.csv
|-- hfl         （预训练模型）
|   `-- chinese-xlnet-base
|-- model.py  （模型文件）
|-- other.py   （没什么用）
|-- output_model （保存的输出文件）
|   |-- 0.pt
|   |-- 1.pt
|   `-- best.pt
|-- readme.md
|-- requirements.txt
|-- test.py         （预测）
`-- train.py        （训练）
```


# 初始配置

## 1、修改`config.py`文件，更改相关参数

## 2、训练

```bash
python train.py
```

## 3、测试

```bash
python test.py
```

