import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TRNNconfig(object):
     """CNN配置参数"""
     def __init__(self):
        self.model_name = 'TextCNN'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = 300
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.embed = 300          # 字向量维度
        self.max_length = 600     #语句最长长度
        self.hidden_dim = 256     # 全连接层神经元
        self.num_classes = 10     # 类别数
        self.dropout = 0.5                                              # 随机失活

        self.learning_rate = 1e-3   # 学习率
        self.batch_size = 256       # mini-batch大小
        self.num_epochs = 300  # 总迭代轮次

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.print_per_batch = 100  # 每多少轮输出一次结果

        self.word2index = dict()
        self.index2word = dict()
        self.category2index = dict()
        self.index2category = dict()
        self.vocab_size = 5000  # 词汇表大小
        self.class_list = list()
        

        self.embedding_pretrained = None
        # self.embedding_pretrained = torch.tensor(
        #     np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
        #     if embedding != 'random' else None                                       # 预训练词向量

class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed, padding_idx=0)

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels=self.config.embed, out_channels=self.config.num_filters, kernel_size=k),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=self.config.max_length-k+1)
                )
                for k in self.config.filter_sizes
            ]
            )
        self.linear1 = nn.Linear(in_features=self.config.num_filters*len(self.config.filter_sizes),out_features=self.config.hidden_dim)
        self.linear2 = nn.Linear(in_features=self.config.hidden_dim,out_features=self.config.num_classes)
        self.embed_dropout = nn.Dropout(self.config.dropout)
        self.linear1_dropout = nn.Dropout(self.config.dropout)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.embed_dropout(embedded)
        embedded = embedded.permute(0,2,1)
        convs_out = [conv(embedded) for conv in self.convs]
        linear1_input = torch.cat(convs_out, dim=1).squeeze(2)
        linear1_input = self.linear1_dropout(linear1_input)
        linear1_output = F.relu(self.linear1(linear1_input))
        linear2_output = self.linear2(linear1_output)
        return linear2_output

if __name__ == "__main__":
    config = TCNNconfig()
    config.max_length = 4
    model = TextCNN(config)

    inputs = np.array(
        [[1,2,3,4],
         [2,2,3,4],
         [3,2,3,4],
         [4,2,3,4],
         [5,2,3,4]
        ]
    )
    inputs = torch.from_numpy(inputs)
    pred = model(inputs)
    print(pred)