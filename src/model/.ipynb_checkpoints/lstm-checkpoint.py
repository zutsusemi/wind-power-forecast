import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.lstm  = nn.GRU(input_size = input_ftr, hidden_size = hidden_ftr, num_layers = 1)
        self.linear = nn.Linear(in_features = hidden_ftr, out_features = output_ftr, bias=True)
    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        # x: (L, B, N)
        padding = torch.rand(self.output_steps, x.shape[1], x.shape[2]).to(self.device) # (S, B, N)
        x = torch.cat((x, padding), dim=0)
        h0 = torch.zeros(1, x.shape[1], self.hidden_ftr).to(self.device)
        # t0 = torch.zeros(6, x.shape[1], self.hidden_ftr).to(self.device)
        out, _  = self.lstm(x, h0) # (L, B?, D*H_out = H_out)
        # h_out = self.linear(out)
        h_out = out.sum(-1).unsqueeze(2)
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
        self.layers = 1+1
        self.enc_first = nn.LSTMCell(input_size= input_ftr, hidden_size = hidden_ftr)
        self.enc = nn.ModuleList([nn.LSTMCell(input_size= hidden_ftr, hidden_size = hidden_ftr) for _ in range(self.layers)])
        self.dec_first = nn.LSTMCell(input_size= hidden_ftr, hidden_size = hidden_ftr)
        self.dec = nn.ModuleList([nn.LSTMCell(input_size= hidden_ftr, hidden_size = hidden_ftr) for _ in range(self.layers)])
        self.linear = nn.Linear(in_features=hidden_ftr, out_features=output_ftr, bias=False)
    def forward(self, x):
        '''
        x: (L, B, H_input)
        '''
        h = [torch.zeros(x.shape[1], self.hidden_ftr).to(self.device) for _ in range(self.layers+1)]
        c = [torch.zeros(x.shape[1], self.hidden_ftr).to(self.device) for _ in range(self.layers+1)]
        # _, (h_out, _)  = self.lstm(x, (h0, t0)) # (L, B?, D*H_out = H_out)
        L, B = x.shape[0], x.shape[1]
        '''
        Encoder
        '''
        for j in range(L):
            h[0], c[0] = self.enc_first(x[j]/10000, (h[0], c[0]))
            xj = h[0]
            for layer, enc in enumerate(self.dec):
                l = layer+1
                h[l], c[l] = enc(xj, (h[l], c[l]))
                xj = h[l]
            
                
        
        '''
        Decoder
        '''
        pred = []
        y = h[-1]
        for j in range(self.output_step):
            h[0], c[0] = self.dec_first(y, (h[0], c[0]))
            xj = h[0]
            for layer, block in enumerate(self.enc):
                l = layer+1
                h[l], c[l] = block(xj, (h[l], c[l]))
                xj = h[l]
            
            pred.append(h[l])
        h_out = self.linear(torch.stack(pred))
        return h_out


class Tre(nn.Module):
    def __init__(self, 
                input_ftr, 
                hidden_ftr, 
                output_ftr,
                input_step=256,
                output_step=144, 
                device='cpu')->None:
        super(Tre, self).__init__()
        self.input_ftr = input_ftr
        self.hidden_ftr  = hidden_ftr
        self.output_step = output_step
        self.input_step = input_step
        self.output_ftr = output_ftr
        self.device  = device
        self.enc_layer = nn.TransformerEncoderLayer(self.hidden_ftr, 8)
        self.enc = nn.TransformerEncoder(self.enc_layer, 3)
        self.proj = nn.Linear(self.input_ftr, self.hidden_ftr)
        self.dec = nn.GRUCell(self.hidden_ftr, self.hidden_ftr)
        self.out = nn.Linear(self.hidden_ftr, self.output_ftr)
        self.linear = nn.Linear(input_step, 1, bias=False)
        self.norm = nn.BatchNorm1d(self.hidden_ftr)
    
    def _pos_embedding(self, embed_dim):
        pos = torch.arange(self.input_step)
        '''
        perform sinosoidal encoding
        '''
        indices = torch.arange(embed_dim//2)
        indices = 1./(10000 ** (indices/embed_dim*2)) # (d,)
        sin_encoding = torch.sin(indices.view(-1, embed_dim//2) * pos.unsqueeze(1))
        cos_encoding = torch.cos(indices.view(-1, embed_dim//2) * pos.unsqueeze(1)) #(t, d)
        enc = torch.cat([sin_encoding, cos_encoding], dim=1)
        enc = enc.unsqueeze(1)
        return enc
    def forward(self, x):
        '''
            x: (L, B, H_input)
        '''
        x = self.proj(x) #(L, B, H_hidden)
        pos = self._pos_embedding(x.shape[-1]).to(self.device)
        x = x + pos
        x = self.enc(x) #(L, B, H_hidden)
        encoding = self.linear(x.permute(1,2,0)).squeeze(2) #(B, H_hidden)
        encoding = self.norm(encoding)
        
        pred = []
        y = torch.zeros((x.shape[-2], x.shape[-1])).to(self.device)
        for j in range(self.output_step):
            encoding = self.dec(y, encoding)
            # encoding = hx + encoding
            y = encoding
            pred.append(y)
        h_out = self.out(torch.stack(pred))
        return h_out



class MLP(nn.Module):
    def __init__(self,
                input_ftr,
                hidden_ftr,
                output_ftr,
                num_sub_features,
                hidden_layer=3
                ):
        super(MLP, self).__init__()
        self.norm_a = nn.BatchNorm1d(num_sub_features)
        self.linear_a = nn.Linear(input_ftr, hidden_ftr)
#         self.norm_h = nn.BatchNorm1d(num_sub_features)
#         self.hidden = nn.Linear(hidden_ftr, hidden_ftr)
#         self.norm = nn.BatchNorm1d(num_sub_features)
        self.linear_o = nn.Linear(hidden_ftr, output_ftr)
        self.relu = nn.ReLU()
#         self.norm_o = nn.BatchNorm1d(num_sub_features)
    def forward(self, x):
#         x = self.norm_a(x)
        x = self.relu(x)
        x = self.linear_a(x)
#         x = self.norm_h(x)
#         x = self.relu(x)
#         x = self.hidden(x)
#         x = self.norm(x)
        x = self.relu(x)
        x = self.linear_o(x)
#         x = self.norm_o(x)
        return x
        
class Tre_trans(nn.Module):
    def __init__(self, 
                input_ftr, 
                hidden_ftr, 
                output_ftr,
                input_step=256,
                output_step=144, 
                device='cpu')->None:
        super(Tre_trans, self).__init__()
        self.input_ftr = input_ftr
        self.hidden_ftr  = hidden_ftr
        self.output_step = output_step
        self.input_step = input_step
        self.output_ftr = output_ftr
        self.device  = device
        self.enc_layer = nn.TransformerEncoderLayer(self.hidden_ftr, 1, dim_feedforward=128)
        self.enc = nn.TransformerEncoder(self.enc_layer, 1)
        self.proj = nn.Linear(self.input_ftr, self.hidden_ftr)
        self.dec = MLP(input_step, 128, output_step, hidden_ftr)
        self.out = nn.Linear(self.hidden_ftr, self.output_ftr)
        self.linear = nn.Linear(hidden_ftr, 1, bias=True)
        self.norm = nn.BatchNorm1d(self.hidden_ftr)
    
    def _pos_embedding(self, embed_dim):
        pos = torch.arange(self.input_step)
        '''
        perform sinosoidal encoding
        '''
        indices = torch.arange(embed_dim//2)
        indices = 1./(10000 ** (indices/embed_dim*2)) # (d,)
        sin_encoding = torch.sin(indices.view(-1, embed_dim//2) * pos.unsqueeze(1))
        cos_encoding = torch.cos(indices.view(-1, embed_dim//2) * pos.unsqueeze(1)) #(t, d)
        enc = torch.cat([sin_encoding, cos_encoding], dim=1)
        enc = enc.unsqueeze(1)
        return enc
    def forward(self, x):
        '''
            x: (B, L, NT, H_input)
        '''
        x = self.proj(x) #(L, B, H_hidden)
        B, L, NT, H = x.shape
        pos = self._pos_embedding(x.shape[-1]).to(self.device)
        x = x + pos
        x = x.permute(1, 0, 2, 3).reshape(self.input_step, -1, self.hidden_ftr) #(T, (B*NT), H)
        x = self.enc(x) #(T, (B*NT), H)
        encoding = x.permute(1,2,0) #((B*NT), H, T)
#         out_pos = self._pos_embedding()
        y = self.dec(encoding).permute(2, 0, 1) #(Tout, (B*NT), H_hidden)
        h_out = self.linear(y).reshape(-1, B, NT, 1)
#         h_out = F.relu(h_out)
#         h_out = (h_out < 2000) * h_out
        return h_out




