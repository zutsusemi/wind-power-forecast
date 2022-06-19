import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, 
                 device: torch.device,
                 input_ftr: int, 
                 hidden_ftr: int, 
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
        self.output_ftr = output_ftr
        
        # construct the model
        self.lstm  = nn.LSTM(input_size = input_ftr, hidden_size = hidden_ftr)
        self.linear = nn.Linear(in_features = hidden_ftr, out_features = output_ftr)
    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(1, x.shape[1], self.hidden_ftr).to(self.device)
        t0 = torch.zeros(1, x.shape[1], self.hidden_ftr).to(self.device)
        _, (h_out, _)  = self.lstm(x, (h0, t0)) # (L, B?, D*H_out = H_out)
        h_out = self.linear(h_out)
        return h_out
