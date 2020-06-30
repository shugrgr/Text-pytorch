import os
import torch
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta
import time

def read_vocab(file_path):
    word2id = dict()
    id2word = dict()
    for number, line in enumerate(open(file_path, 'r')):
        text = line.strip('\n')
        word2id[text] = number
        id2word[number] = text
    word2id['<UNK>'] = number + 1
    id2word[number + 1] = '<UNK>'
    return word2id, id2word
    

class myDataset(Dataset):
    def __init__(self, path, config):
        file = open(path, 'r')
        self.x_data = list()
        self.y_data = list()
       
        for line in file:
            texts = line.strip().split('\t')
            label = texts[0]
            text = texts[1] if len(texts[1]) <= config.max_length else texts[1][:config.max_length]

            self.y_data.append(config.category2index[label])
            x_tmp = list()
            for char in text:
                if char not in config.word2index:
                    x_tmp.append(config.word2index["<UNK>"])
                else:
                    x_tmp.append(config.word2index[char])
            if len(x_tmp) < config.max_length:
                x_tmp.extend([0 for i in range(config.max_length - len(x_tmp))])
            self.x_data.append(np.array(x_tmp))
        self.x_data = torch.from_numpy(np.array(self.x_data))
        self.y_data = torch.from_numpy(np.array(self.y_data))
        self.len = self.x_data.size(0)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))