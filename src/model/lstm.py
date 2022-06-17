import torch
import numpy as np
import torch.nn as nn

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