import numpy as np
import keras.preprocessing.text as T
import preprocessor as p 
import re
import wordninja
import copy
import random
import gensim.models.keyedvectors as word2vec
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import gc


sequence_length = 128
lstm_hidden_size = 512
lstm_dropout = 0.2
net_dropout = 0.3
linear_size = 128
in_channels = 1
out_channels = 25
out_channels2 = 25
kernel_size = [(2,1024),(3,1024),(4,1024),(5,1024)]
kernel_size2 = [(2,1024),(3,1024),(4,1024),(5,1024)]
kernel_size5 = (1,1024)


learning_rate = 1e-5
weight_decay = 4e-5
batch_size = 32
total_epoch = 120
iter_num = 5

# PGCNN
# Paper "Parameterized Convolutional Neural Networks for Aspect Level Sentiment Classification"
# https://www.aclweb.org/anthology/D18-1136.pdf
def get_PGCNN():
    def kmax_pooling(x, dim, k):

        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]

        return x.gather(dim, index)

    class LSTMClassifier(nn.Module):
        def __init__(self):
            super(LSTMClassifier, self).__init__()
            
            self.model_name = 'PGCNN'
        
            self.dropout = nn.Dropout(net_dropout)
            
            self.k = 1
            self.linear = nn.Linear(out_channels*4*self.k, linear_size)
            self.out = nn.Linear(linear_size, 3)
            self.relu = nn.ReLU()
            
            self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=K) for K in kernel_size])
            self.conv4 = nn.Conv2d(in_channels=in_channels,out_channels=25*1024*(2+3+4+5),kernel_size=kernel_size5)
            
        def forward(self, x, x_len, epoch, target_word, _):
            
            b_size = x.shape[0]
            x_tp = x.transpose(0, 1)
            
            # 16x1(channel)x50x300 -> 16x25(filter_num)x?x1 
            conved = [conv(x).squeeze(3) for conv in self.convs]
            
            conved4 = self.relu(self.conv4(target_word).squeeze(3)) # 16x105000x?
            pooled4 = F.avg_pool1d(conved4,conved4.shape[2]).squeeze(2) # 16x105000
            sep = pooled4.view(b_size*out_channels,1,-1,1024) # 16x(25x1x(2+3+4+5)x1024)
            sep = [sep[:,:,:2,:],sep[:,:,2:5,:],sep[:,:,5:9,:],sep[:,:,9:,:] ]
            sig = [torch.sigmoid(F.conv2d(x_tp,sep_filter,groups=b_size).squeeze(3).view(b_size,out_channels,-1)) for sep_filter in sep]
            
            conved_gate = [i*j for i,j in zip(conved,sig)]
            pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved_gate]
            cat = torch.cat(pooled, dim=1) 
        
            linear = self.relu(self.linear(cat))
            linear = self.dropout(linear)
            out = self.out(linear)
            
            return out

    model = LSTMClassifier()
    return model


# # GCAE Model
# # Paper "Aspect Based Sentiment Analysis with Gated Convolutional Networks"
# # https://www.aclweb.org/anthology/P18-1234.pdf
def get_GCAE():
    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index)


    class LSTMClassifier(nn.Module):

        def __init__(self):

            super(LSTMClassifier, self).__init__()
            
            self.model_name = 'GCAE_Model'

            self.dropout = nn.Dropout(net_dropout)
            
            self.k = 1
            self.linear = nn.Linear(out_channels*4, linear_size)
            self.out = nn.Linear(linear_size, 3)
            self.relu = nn.ReLU()
            self.linear5 = nn.Linear(out_channels*8*self.k, out_channels*4*self.k)
            
            self.convs_tanh = nn.ModuleList([nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=K) for K in kernel_size])
            self.convs_relu = nn.ModuleList([nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=K) for K in kernel_size])
            self.V = nn.Parameter(torch.rand([1024,out_channels],requires_grad=True).cuda())
            self.conv4 = nn.Conv2d(in_channels=in_channels,out_channels=1024,kernel_size=kernel_size5)
            
        def forward(self, x, x_len, epoch, target_word, _):
            
            target_word = target_word.squeeze(1)
            if target_word.size(1) != 300:
                target_word = target_word.sum(1) / target_word.size(1)
        
            # 16x1(channel)x50x300 -> 16x25(filter_num)x?x1
            conved_tanh = [F.tanh(conv(x).squeeze(3)) for conv in self.convs_tanh]
            conved_relu = [self.relu(conv(x).squeeze(3)+torch.mm(target_word,self.V).unsqueeze(2)) for conv in self.convs_relu]
            conved_mul = [i*j for i,j in zip(conved_tanh,conved_relu)]
            pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved_mul]
            cat = torch.cat(pooled, dim=1) 
            
            linear = self.relu(self.linear(cat))
            linear = self.dropout(linear)
            out = self.out(linear)
            
            return out

    model = LSTMClassifier()
    return model


# Kim CNN Model
def get_Kim_CNN():
    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index)


    class LSTMClassifier(nn.Module):

        def __init__(self):

            super(LSTMClassifier, self).__init__()
            
            self.model_name = 'Kim_CNN'
            
            self.dropout = nn.Dropout(net_dropout)
            
            self.k = 1
            self.linear = nn.Linear(out_channels*4, linear_size)
            self.out = nn.Linear(128, 3)
            self.relu = nn.ReLU()
            
            self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=K) for K in kernel_size])
            
        def forward(self, x, x_len, epoch, target_word, _):
            
            # 16x1(channel)x50x300 -> 16x100(filter_num)x?x1 
            conved = [self.relu(conv(x).squeeze(3)) for conv in self.convs]
            pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved]
            cat = torch.cat(pooled, dim=1) 
            
            linear = self.relu(self.linear(cat))
            linear = self.dropout(linear)
            out = self.out(linear)
            
            return out

    model = LSTMClassifier()
    return model


# # BiLSTM
def get_BiLSTM():
    class LSTMClassifier(nn.Module):

        def __init__(self):

            super(LSTMClassifier, self).__init__()
            
            self.model_name = 'BiLSTM'
            
            self.dropout = nn.Dropout(net_dropout)
            
            self.hidden_size = lstm_hidden_size
            self.lstm = nn.LSTM(1024, self.hidden_size, dropout=lstm_dropout, bidirectional=True)
            self.linear = nn.Linear(self.hidden_size*2, linear_size)
            self.out = nn.Linear(linear_size, 3)
            self.relu = nn.ReLU()
            
        def forward(self, x, x_len, epoch, target_word, _):
            
            x = x.squeeze(1)
            
            seq_lengths, perm_idx = x_len.sort(0, descending=True)
            seq_tensor = x[perm_idx,:,:]
            packed_input = pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)
            packed_output, (ht, ct) = self.lstm(packed_input)
            _, unperm_idx = perm_idx.sort(0)
            h_t = ht[:,unperm_idx,:]
            h_t = torch.cat((h_t[0,:,:self.hidden_size], h_t[1,:,:self.hidden_size]), 1)
            
            linear = self.relu(self.linear(h_t))
            linear = self.dropout(linear)
            out = self.out(linear)
            
            return out

    model = LSTMClassifier()
    return model


# # TAN
# # Paper "Stance Classification with Target-Specific Neural Attention Networks"
# # https://www.ijcai.org/Proceedings/2017/0557.pdf

def get_TAN():
    def Attention_Stance(hidden_unit, h_embedding2, W_h, b_tanh, length, epoch, x, target_word):
        word_tensor = torch.zeros(h_embedding2.size(0),h_embedding2.size(1),1024).cuda() #4x128x300
        word_tensor[:,:,:] = target_word
        word_tensor = torch.cat((h_embedding2,word_tensor),2) #16x50x600
        
        s1 = h_embedding2.size(0) # batch size 4
        s2 = h_embedding2.size(1) # time step 128
        
        # 16x50x600 x 600x1 + 50x1 = 16x50x1
        m1 = torch.mm(word_tensor.view(-1,2048),W_h).view(s1, s2, -1)
        u = (m1 + b_tanh.unsqueeze(1)).squeeze(2) #16x50x1 -> 16x50
        
        for i in range(len(length)):
            u[i,length[i]:] = torch.Tensor([-1e6])
        # alphas size 16x50
        alphas = nn.functional.softmax(u)        

        # context size 16x1x50 x 16x50x300 = 16x1x300 = 16x300
        context = torch.bmm(alphas[:,:hidden_unit.size(1)].unsqueeze(1), hidden_unit).squeeze(1)
        return context, alphas

    class LSTMClassifier(nn.Module):

        def __init__(self):
            super(LSTMClassifier, self).__init__()
            self.model_name = 'TAN'
            self.dropout = nn.Dropout(net_dropout)
            self.hidden_size = lstm_hidden_size
            self.lstm = nn.LSTM(1024, self.hidden_size, dropout=lstm_dropout, bidirectional=True) 
            self.linear = nn.Linear(self.hidden_size*2, linear_size)
            self.out = nn.Linear(linear_size, 3)
            self.relu = nn.ReLU()
            
            self.W_h = nn.Parameter(torch.rand([2*1024,1],requires_grad=True))
            self.b_tanh = nn.Parameter(torch.rand(sequence_length,requires_grad=True))
            
        def forward(self, x, x_len, epoch, target_word, tokens):       
            x = x.squeeze(1)
            target_word = target_word.squeeze(1)
            if target_word.size(1) != 1024:
                target_word = target_word.sum(1) / target_word.size(1)
            target_word = target_word.unsqueeze(1)
            
            seq_lengths, perm_idx = x_len.sort(0, descending=True)
            seq_tensor = x[perm_idx,:,:]
            packed_input = pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)
            packed_output, (ht, ct) = self.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output,batch_first=True)
            _, unperm_idx = perm_idx.sort(0)
            h_lstm = output[unperm_idx,:,:]
            
            atten, alpha = Attention_Stance(h_lstm,x,self.W_h,self.b_tanh,x_len,epoch,tokens, target_word)
            atten = self.dropout(atten)
            
            linear = self.relu(self.linear(atten))
            linear = self.dropout(linear)
            out = self.out(linear)
            
            return out

    model = LSTMClassifier()
    return model


# # ATGRU
# # Paper "Connecting targets to tweets: Semantic attention-based model for target-specific stance detection"
# # http://dro.dur.ac.uk/25714/1/25714.pdf

def get_ATGRU():
    def Attention_Stance(hidden_unit, W_h, W_z, b_tanh, v, length, epoch, x, target_word):
        
        # hidden_size 150, hidden_unit 16x50x300
        s1 = hidden_unit.size(0) # batch size 16
        s2 = hidden_unit.size(1) # time step ?
        s3 = hidden_unit.size(2) # hidden dimension 150x2
        
        word_tensor = torch.zeros(s1,s2,1024).cuda() #16x?x300
        word_tensor[:,:,:] = target_word
        
        # 16x?x300 x 300x300 + 16x?x300 x 300x300 + 1x300 = 16x?x300+16x1x300+1x300 = 16x?x300
        m1 = torch.mm(hidden_unit.view(-1,hidden_unit.size(2)),W_h).view(-1, s2, s3)
        m2 = torch.mm(word_tensor.view(-1,1024),W_z).view(-1, s2, s3)
        sum_tanh = torch.tanh(m1 + m2 + b_tanh.unsqueeze(0))
        # sum_tanh*v = 16x?x300 x 300x1 = 16x?x1 = 16x?
        u = torch.mm(sum_tanh.view(-1,s3),v.unsqueeze(1)).view(-1,s2,1).squeeze(2)
        
        for i in range(len(length)):
            u[i,length[i]:] = torch.Tensor([-1e6])
        # alphas size 16x?
        alphas = nn.functional.softmax(u)        

        # context size 16x1x? x 16x?x300 = 16x1x300 = 16x300
        context = torch.bmm(alphas.unsqueeze(1), hidden_unit).squeeze(1)

        return context, alphas


    class LSTMClassifier(nn.Module):

        def __init__(self):

            super(LSTMClassifier, self).__init__()
            
            self.model_name = 'ATGRU'
            
            self.dropout = nn.Dropout(net_dropout)
            
            self.hidden_size = lstm_hidden_size
            self.lstm = nn.LSTM(1024, self.hidden_size, dropout=lstm_dropout, bidirectional=True) 
            self.linear = nn.Linear(self.hidden_size*2, linear_size)
            self.out = nn.Linear(linear_size, 3)
            self.relu = nn.ReLU()
            
            self.W_h = nn.Parameter(torch.rand([self.hidden_size*2,self.hidden_size*2],requires_grad=True))
            self.W_z = nn.Parameter(torch.rand([self.hidden_size*2,self.hidden_size*2],requires_grad=True))
            self.b_tanh = nn.Parameter(torch.rand(self.hidden_size*2,requires_grad=True))
            self.v = nn.Parameter(torch.rand(self.hidden_size*2,requires_grad=True))
            
        def forward(self, x, x_len, epoch, target_word, tokens):
            
            x = x.squeeze(1)
            target_word = target_word.squeeze(1)
            if target_word.size(1) != 1024:
                target_word = target_word.sum(1) / target_word.size(1)
            target_word = target_word.unsqueeze(1)
            
            seq_lengths, perm_idx = x_len.sort(0, descending=True)
            seq_tensor = x[perm_idx,:,:]
            packed_input = pack_padded_sequence(seq_tensor, seq_lengths,batch_first=True)
            packed_output, (ht, ct) = self.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output,batch_first=True)
            _, unperm_idx = perm_idx.sort(0)
            h_lstm = output[unperm_idx,:,:]
            
            atten, alpha = Attention_Stance(h_lstm,self.W_h,self.W_z,self.b_tanh,self.v,x_len,epoch,tokens,target_word)
            atten = self.dropout(atten)
            
            linear = self.relu(self.linear(atten))
            linear = self.dropout(linear)
            out = self.out(linear)
            
            return out

    model = LSTMClassifier()
    return model


def compute_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.nn.functional.softmax(preds)
    _, indices = torch.max(rounded_preds, 1)
    rounded_preds = indices
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    
    y_pred = np.array(rounded_preds.cpu().numpy())
    y_true = np.array(y.cpu().numpy())
    result = precision_recall_fscore_support(y_true, y_pred, average=None,labels=[0,1,2])
    f1_average = (result[2][0]+result[2][2])/2
        
    return acc, f1_average, result[0], result[1], result[2]
