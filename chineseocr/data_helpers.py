# coding:utf-8
import numpy as np
import re
import itertools
from collections import Counter
import importlib, sys
importlib.reload(sys)

# 剔除英文的符号
def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(open_data_file, close_data_file, lure_data_file):
    """
    加载三分类训练数据，为数据打上标签
    (X,[0,0,1]
    """

    open_examples = list(open(open_data_file, "r", encoding='utf-8').readlines())
    open_examples = [s.strip() for s in open_examples]
    close_examples = list(open(close_data_file, "r", encoding='utf-8').readlines())
    close_examples = [s.strip() for s in close_examples]
    lure_examples = list(open(lure_data_file, "r", encoding='utf-8').readlines())
    lure_examples = [s.strip() for s in lure_examples]

    x_text = open_examples + close_examples + lure_examples

    x_text = [sent for sent in x_text]
    # 定义类别标签 ，格式为one-hot的形式: y=1--->[0,1,0]
    open_labels = [[1, 0, 0] for _ in open_examples]
    close_labels = [[0, 1, 0] for _ in close_examples]
    lure_labels = [[0, 0, 1] for _ in lure_examples]
    y = np.concatenate([open_labels, close_labels, lure_labels], 0)
    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            # 随机产生以一个乱序数组，作为数据集数组的下标
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 划分批次
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# 测试代码用的
if __name__ == '__main__':
    open_data_file = './fenci/open.txt'
    close_data_file = './fenci/close.txt'
    lure_data_file = './fenci/lure.txt'
    load_data_and_labels(open_data_file, close_data_file, lure_data_file)








