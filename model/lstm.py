import torch
import numpy as np
import torch.nn as nn

__all__ = ('model_lstm', 'Seq2SeqLSTM')

class model_lstm(nn.Module):
    def __init__(self, input_ftr, hidden_ftr, output_ftr, device):
        super(model_lstm, self).__init__()
        self.input_ftr = input_ftr
        self.hidden_ftr  = hidden_ftr
        self.output_ftr = output_ftr
        self.device  = device
        self.lstm  = nn.LSTM(input_size= input_ftr, hidden_size = hidden_ftr)
        self.linear = nn.Linear(in_features=hidden_ftr, out_features=output_ftr)
    def forward(self, x):
        h0 = torch.zeros(1, x.shape[1], self.hidden_ftr).to(self.device)
        t0 = torch.zeros(1, x.shape[1], self.hidden_ftr).to(self.device)
        _, (h_out, _)  = self.lstm(x, (h0, t0)) # (L, B?, D*H_out = H_out)
        h_out = self.linear(h_out)
        return h_out
    # def inference(self, x):
    #     return 1

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_ftr, hidden_ftr, output_ftr, output_step=1, device='cpu'):
        super(Seq2SeqLSTM, self).__init__()
        self.input_ftr = input_ftr
        self.hidden_ftr  = hidden_ftr
        self.output_step = output_step
        self.output_ftr = output_ftr
        self.device  = device
        self.enc = nn.LSTMCell(input_size= input_ftr, hidden_size = hidden_ftr)
        self.dec = nn.LSTMCell(input_size= hidden_ftr, hidden_size = hidden_ftr)
        self.linear = nn.Linear(in_features=hidden_ftr, out_features=output_ftr)
    def forward(self, x):
        '''
        x: (L, B, H_input)
        '''
        h = torch.zeros(x.shape[1], self.hidden_ftr).to(self.device)
        c = torch.zeros(x.shape[1], self.hidden_ftr).to(self.device)
        # _, (h_out, _)  = self.lstm(x, (h0, t0)) # (L, B?, D*H_out = H_out)
        L, B = x.shape[0], x.shape[1]
        '''
        Encoder
        '''
        for j in range(L):
            h, c = self.enc(x[j], (h, c)) # B, H
        
        '''
        Decoder
        '''
        pred = []
        y = h
        for j in range(self.output_step):
            h, c = self.dec(y, (h, c))
            y = h
            pred.append(self.linear(y)) # B, H_out
        h_out = torch.stack(pred)
        return h_out
