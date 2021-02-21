import os
import sys
import time
import logging
import argparse
import gensim
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from gensim.models import KeyedVectors
from collections import OrderedDict

sys.path.append('../')

from utils import data_helpers as dh
from utils import param_parser as parser

logging.getLogger('Pytorch').disabled = True

class HARNN(nn.Module):
    def __init__(self, args, device):
        super(HARNN, self).__init__()
        #双向 lstm
        self.alpha = args.alpha
        self.device = device
        self.num_classes_list = args.num_classes_list
        self.num_layers = args.lstm_layers
        self.num_directions = 2
        self.batch_size = args.batch_size
        self.attention_unit_size = args.attention_dim
        self.lstm_hidden_size = args.lstm_dim
        self.fc_hidden_size = args.fc_dim
        self.output_size = args.pad_seq_len * args.batch_size * self.num_directions * self.lstm_hidden_size
        self.Bi_lstm = nn.LSTM(input_size=args.embedding_dim,
                               hidden_size=self.lstm_hidden_size,
                               num_layers=self.num_layers,
                               batch_first=True,
                               bidirectional=True)
        
        #初始 h_0, c_0
        self.hidden_0 = torch.randn(self.num_layers * self.num_directions, self.batch_size, self.lstm_hidden_size).to(device)
        self.memory_0 = torch.randn(self.num_layers * self.num_directions, self.batch_size, self.lstm_hidden_size).to(device)
    
    def forward(self, input_x, input_x_1, input_x_2, input_x_3, input_x_4, total_classes):
        def _attention(input_x, num_classes):
            '''
            Args:
                input_x: [batch_size, sequence_length, lstm_hidden_size * 2]
                num_classes: The number of i th level classes.
            Returns:
                attention_weight: [batch_size, num_classes, sequence_length]
                attention_out: [batch_size, lstm_hidden_size * 2]
            '''
            num_units = input_x.size()[-1]
            W_s1 = torch.randn([self.attention_unit_size, num_units], requires_grad=True).to(self.device)
            W_s2 = torch.randn([num_classes, self.attention_unit_size], requires_grad=True).to(self.device)
            
            # attention_matrix = [batch_size, num_classes, sequence_length]
            attention_matrix = torch.matmul(W_s2, F.tanh(torch.matmul(W_s1, torch.transpose(input_x, 1, 2))))
            attention_weight = F.softmax(attention_matrix)
            
            # attention_out = [batch_size, num_classes, lstm_hidden_size * 2]
            attention_out = torch.matmul(attention_weight, input_x)

            # attention_out = [batch_size, lstm_hidden_size * 2]
            attention_out = torch.mean(attention_out, dim=1, keepdim=False)
            return attention_weight, attention_out

        def _fc_layer(input_x):
            """
            Args:
                input_x: [batch_size, *]
                name: Scope name.
            Returns:
                fc_out: [batch_size, fc_hidden_size]
            """
            num_units = input_x.size()[-1]
            W = torch.randn([num_units, self.fc_hidden_size], requires_grad=True).to(self.device)
            return F.relu(torch.matmul(input_x, W))

        def _local_layer(input_x, input_att_weigh, num_classes):
            """
            Args:
                input_x: [batch_size, fc_hidden_size]
                input_att_weight: [batch_size, num_classes, sequence_length]
                num_classes: Number of classes.
            Returns:
                logits: [batch_size, num_classes]
                scores: [batch_size, num_classes]
                visual: [batch_size, sequence_length]
            """
            num_units = input_x.size()[-1]
            #fc = nn.Linear(num_units, num_classes, bias=True)
            W = torch.randn([num_units, num_classes], requires_grad=True).to(self.device)

            #logits = [batch_size, num_classes]
            logits = torch.matmul(input_x, W).to(self.device)
            scores = F.sigmoid(logits).to(self.device)

            visual = torch.mul(input_att_weigh, torch.unsqueeze(scores, -1)).to(self.device)
            visual = F.softmax(visual).to(self.device)

            # visual = [batch_size, sequence_length]
            visual = torch.mean(visual, dim=1, keepdim=False).to(self.device)
            return logits, scores, visual

        input_x = input_x.to(self.device)
        self.input_x_1 = input_x_1.to(self.device)
        self.input_x_2 = input_x_2.to(self.device)
        self.input_x_3 = input_x_3.to(self.device)
        self.input_x_4 = input_x_4.to(self.device)
        self.total_classes = total_classes.to(self.device)

        #outputs = [batch_size, pad_seq_len, lstm_hidden_size * 2]
        self.lstm_outputs, (h_n, c_n) = self.Bi_lstm(input_x, (self.hidden_0, self.memory_0))
        self.lstm_outputs_pool = torch.mean(self.lstm_outputs, dim=1, keepdim=False).to(self.device)

        # First Level
        self.first_att_weight, self.first_att_out = _attention(self.lstm_outputs, self.num_classes_list[0])
        self.first_local_input = torch.cat([self.lstm_outputs_pool, self.first_att_out], dim=1)
        self.first_local_fc_out = _fc_layer(self.first_local_input)
        self.first_logits, self.first_scores, self.first_visual = _local_layer(self.first_local_fc_out, self.first_att_weight, self.num_classes_list[0])

        # Second Level
        self.second_att_input = torch.mul(self.lstm_outputs, torch.unsqueeze(self.first_visual, -1))
        self.second_att_weight, self.second_att_out = _attention(self.second_att_input, self.num_classes_list[1])
        self.second_local_input = torch.cat([self.lstm_outputs_pool, self.second_att_out], dim=1)
        self.second_local_fc_out = _fc_layer(self.second_local_input)
        self.second_logits, self.second_scores, self.second_visual = _local_layer(self.second_local_fc_out, self.second_att_weight, self.num_classes_list[1])

        # Third Level
        self.third_att_input = torch.mul(self.lstm_outputs, torch.unsqueeze(self.second_visual, -1))
        self.third_att_weight, self.third_att_out = _attention(self.third_att_input, self.num_classes_list[2])
        self.third_local_input = torch.cat([self.lstm_outputs_pool, self.third_att_out], dim=1)
        self.third_local_fc_out = _fc_layer(self.third_local_input)
        self.third_logits, self.third_scores, self.third_visual = _local_layer(self.third_local_fc_out, self.third_att_weight, self.num_classes_list[2])

        # Fourth Level
        self.fourth_att_input = torch.mul(self.lstm_outputs, torch.unsqueeze(self.third_visual, -1))
        self.fourth_att_weight, self.fourth_att_out = _attention(self.fourth_att_input, self.num_classes_list[3])
        self.fourth_local_input = torch.cat([self.lstm_outputs_pool, self.fourth_att_out], dim=1)
        self.fourth_local_fc_out = _fc_layer(self.fourth_local_input)
        self.fourth_logits, self.fourth_scores, self.fourth_visual = _local_layer(self.fourth_local_fc_out, self.fourth_att_weight, self.num_classes_list[3])

        self.ham_out = torch.cat([self.first_local_fc_out, self.second_local_fc_out,
                                  self.third_local_fc_out, self.fourth_local_fc_out], dim=1)
        
        # fc_out = [batch_size, fc_hidden_size]
        self.fc_out = _fc_layer(self.ham_out).to(device)

        # Global scores
        self.global_num_units = self.fc_out.size()[-1]
        W = torch.randn([self.global_num_units, args.total_classes], requires_grad=True).to(self.device)
        self.global_logits = torch.matmul(self.fc_out, W)
        #self.global_scores = F.sigmoid(self.global_logits).to(self.device)

        # scores
        #self.local_scores = torch.cat([self.first_scores, self.second_scores, self.third_scores, self.fourth_scores], dim=1).to(self.device)
        #self.scores = torch.add(self.alpha * self.local_scores, (1.0 - self.alpha) * self.global_scores)

        return self.first_logits.to(self.device), self.second_logits.to(self.device), \
               self.third_logits.to(self.device), self.fourth_logits.to(self.device), self.global_logits
        '''
        # loss
        self.loss_func = nn.CrossEntropyLoss()

        self.loss = self.loss_func(self.first_logits, self.input_x_1) + self.loss_func(self.second_logits, self.input_x_2) + \
                    self.loss_func(self.third_logits, self.input_x_3) + self.loss_func(self.fourth_logits, self.input_x_4) + \
                    self.loss_func(self.global_logits, self.total_classes)
        '''

class TextData(torch.utils.data.Dataset):
    def __init__(self, args, input_file, word2idx, embedding_matrix):   # read input_file
        if not input_file.endswith('.json'):
            raise IOError("[Error] The research record is not a json file. "
                          "Please preprocess the research record into the json file.")
        
        self.Data = dh.load_data_and_labels(args, input_file, word2idx)

        # 下面将通过 embedding_matrix 将索引列表转化为向量列表, 将每个索引替换为一个 1xN 向量
        def _index_to_vector(x, embedding_matrix):	# x 为一行单词对应的索引列表
            result = []
            for index in x:
                result.append(embedding_matrix[int(index)])
            return result
        
        self.Data['content_vector'] = []
        for content in self.Data['content_index']:
            self.Data['content_vector'].append(_index_to_vector(content, embedding_matrix))	# 新建一个列表, 用于存向量列表
        
    
    def __len__(self):
        return len(self.Data['content_index'])
    
    def __getitem__(self, idx):
        return torch.Tensor(self.Data['content_vector'][idx]), torch.Tensor(self.Data['section'][idx]), \
               torch.Tensor(self.Data['subsection'][idx]), torch.Tensor(self.Data['group'][idx]), \
               torch.Tensor(self.Data['subgroup'][idx]), torch.Tensor(self.Data['onehot_labels'][idx])

def train(args, model, train_loader, device, epoch):
    model.train()	# ?
    for batch_idx, (x, section, subsection, group, subgroup, onehot_labels) in enumerate(train_loader):
        x = x.to(device)
        section = section.to(device)
        subsection = subsection.to(device)
        group = group.to(device)
        subgroup = subgroup.to(device)
        onehot_labels = onehot_labels.to(device)

        optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
        optimizer.zero_grad()

        first_logits, second_logits, third_logits, fourth_logits, global_logits = model(x, section, subsection, group, subgroup, onehot_labels)	# 待修改

        loss_func = nn.MSELoss()
        loss = loss_func(first_logits, section) + loss_func(second_logits, subsection) + \
               loss_func(third_logits, group) + loss_func(fourth_logits, subgroup)# + \
               #loss_func(global_logits, onehot_labels)
        loss.backward(retain_graph=True)

        optimizer.step()
        # 打印进度条

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(x), len(train_loader.dataset),	#为什么 len(train_loader.dataset) 恒为 128？
                   100. * batch_idx / len(train_loader), loss.item()))
    print("\n\n\n")

def train_HARNN(args, device):
    # Load word2vec model
    print("Loading data...")
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Load sentences, labels, and training parameters
    print("Data processing...")
    train_data = TextData(args, args.train_file, word2idx, embedding_matrix)
    #	test_data = TextData(args, args.test_file, word2idx, embedding_matrix)
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True, num_workers=1)
    #	test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False, num_workers=1)

    model = HARNN(args, device).to(device)
    print(model)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, device, epoch)

if __name__ == '__main__':
    args = parser.parameter_parser()	# add parser by using argparse module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	#使用 gpu
    train_HARNN(args, device)
    x = print("Press any key to continue...")