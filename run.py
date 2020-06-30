import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cnn_model import TCNNconfig, TextCNN
from rnn_model import TRNNconfig, TextRNN
from utils import read_vocab, myDataset, get_time_dif
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import time
from tensorboardX import SummaryWriter
from sklearn import metrics

def train(checkpoint_path, model, config):
    start_time = time.time()

    model = nn.DataParallel(model, device_ids=[0,1])

    train_dataset = myDataset('/search/odin/wts/my-pytorch-try/text_cnn_rnn/data/cnews/cnews.train.txt', config)
    val_dataset = myDataset('/search/odin/wts/my-pytorch-try/text_cnn_rnn/data/cnews/cnews.val.txt', config)
    train_loader= DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader= DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    dev_best_acc = 0.0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=checkpoint_path + '/log/' + time.strftime('%m-%d_%H.%M', time.localtime()))
   

    for epoch in range(config.num_epochs):
        for i, data in enumerate(val_loader, 0):#python enumerate用法总结
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            outputs = model(inputs)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                golden_tag = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(golden_tag, predict)
                dev_acc, dev_loss = evaluate(config, model, val_loader)
                if dev_acc > dev_best_acc and dev_loss < dev_best_loss:
                    dev_best_acc = dev_acc
                    dev_best_loss = dev_loss
                    torch.save(model.module.state_dict(), checkpoint_path+'/cnn_model.pkl')
                    last_improve = total_batch
                    improve = '*'
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Epoch: {0:>6}, Iter: {1:>6},  Train Loss: {2:>5.2},  Train Acc: {3:>6.2%},  Val Loss: {4:>5.2},  Val Acc: {5:>6.2%},  Time: {6} {7}'
                print(msg.format(epoch, total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    #test(config, model, test_iter)

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(labels_all), report, confusion
    return acc, loss_total / len(labels_all)

def test(checkpoint_path, model, config):
    test_dataset = myDataset('/search/odin/wts/my-pytorch-try/text_cnn_rnn/data/cnews/cnews.test.txt', config)
    test_loader= DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    
    
    model.load_state_dict(torch.load(checkpoint_path+'/cnn_model.pkl'))
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            outputs = model(inputs)

            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)

    report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)

    print(acc)
    print(report)
    print(confusion)
    return acc, loss_total / len(labels_all), report, confusion





if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("""usage: python run_cnn.py [train / test] [cnn/rnn]""")

    if sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test] [cnn/rnn]""")
    
    if sys.argv[2] not in ['cnn','rnn']:
        raise ValueError("""usage: python run_cnn.py [train / test] [cnn/rnn]""")

    print('Configuring {0} model...'.format(sys.argv[2]))
    model_name = sys.argv[2]
    if model_name == 'cnn':
        config = TCNNconfig()
    elif model_name == 'rnn':
        config = TRNNconfig()
    
    word2id, id2word = read_vocab("/search/odin/wts/my-pytorch-try/text_cnn_rnn/data/cnews/cnews.vocab.txt")
    config.word2index = word2id
    config.index2word = id2word
    config.vocab_size = len(word2id)  # 词汇表大小
    config.category2index = {"财经":0,"房产":1,"家居":2,"教育":3,"科技":4,"时尚":5,"时政":6,"体育":7,"游戏":8,"娱乐":9}
    config.index2category = {0:"财经",1:"房产",2:"家居",3:"教育",4:"科技",5:"时尚",6:"时政",7:"体育",8:"游戏",9:"娱乐"}
    config.class_list = ["财经","房产","家居","教育","科技","时尚","时政","体育","游戏","娱乐"]
    
    if model_name == 'cnn':
        model = TextCNN(config)
    elif model_name == 'rnn':
        model = TextRNN(config)
    model.to(config.device)

    checkpoint_path = "/search/odin/wts/my-pytorch-try/text_cnn_rnn/data/checkpoint/{}".format(model_name)
    if sys.argv[1] == 'train':
        train(checkpoint_path, model, config)
    else:
        test(checkpoint_path,model, config)