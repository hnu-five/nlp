##关于模型的更新1.0
模型训练完成

main.py训练文件

evaluate.py模型使用文件

data.json模型训练集

## 更新2.0

1. main的tag编码：上一版将姓名合二为一，造成了main和evaluate的模型规格不一样，不能直接使用。
2. 修改了evaluate，使得能够输入text，直接输出对应特征
3. 增加了test.json，用于evaluate测试
4. 增加了ciku.json，用于保存训练过的词，方便测试使用