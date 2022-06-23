import torch
import torch.nn as nn

__all__ = ('LSTM', 'Seq2SeqLSTM')

class LSTM(nn.Module):
    def __init__(self, 
                 device: torch.device,
                 input_ftr: int, 
                 hidden_ftr: int, 
                 output_steps: int,
                 output_ftr: int) -> None:
        """constructor

        Args:
            device (torch.device): the device to use
            input_ftr (int): the number of input features
            hidden_ftr (int): the number of hidden features
            output_ftr (int): the number of output features 
        
        Returns:
            None
        """
        super(LSTM, self).__init__()
        self.device = device
        self.input_ftr = input_ftr
        self.hidden_ftr  = hidden_ftr
        self.output_steps = output_steps
        self.output_ftr = output_ftr
        
        # construct the model
        self.lstm  = nn.GRU(input_size = input_ftr, hidden_size = hidden_ftr, num_layers = 6)
        self.linear = nn.Linear(in_features = hidden_ftr, out_features = output_ftr)
    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        # x: (L, B, N)
        padding = torch.zeros(self.output_steps, x.shape[1], x.shape[2]).to(self.device) # (S, B, N)
        x = torch.cat((x, padding), dim=0)
        h0 = torch.zeros(6, x.shape[1], self.hidden_ftr).to(self.device)
        # t0 = torch.zeros(6, x.shape[1], self.hidden_ftr).to(self.device)
        out, _  = self.lstm(x, h0) # (L, B?, D*H_out = H_out)
        h_out = self.linear(out)
        h_out = h_out[- self.output_steps:, :, :]
        return h_out



class Seq2SeqLSTM(nn.Module):
    def __init__(self, 
                input_ftr, 
                hidden_ftr, 
                output_ftr, 
                output_step=1, 
                device='cpu') -> None:
        """constructor

        Args:
            device (torch.device): the device to use
            input_ftr (int): the number of input features
            hidden_ftr (int): the number of hidden features
            output_ftr (int): the number of output features
            ouput_step (int): the steps to predict
        
        Returns:
            None
        """
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
