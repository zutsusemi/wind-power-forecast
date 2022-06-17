import torch
from torch.utils.data import DataLoader
from dataset.data import Scada
from utils.split import DatasetSampler
import torch.nn.functional as f
from model.lstm import model_lstm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device "+str(device)+".")

path = './train_val.csv'

batchsize = 80
num_epochs = 500
lr = 0.01

dataset = Scada(path)
sampler = DatasetSampler(len(dataset), len(dataset) // 5)
train, val = sampler(dataset)
train_loader = DataLoader(train, batchsize, shuffle=True)
val_loader = DataLoader(val, 1, shuffle=True)
model = model_lstm(3, 64, 1, device=device).to(device)
model.to(device)
loss = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

class trainer:
    def __init__(self, train_loader, val_loader, model, loss, optm, num_e, device):
        self.model = model
        self.train_loader  = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss, self.optm = loss, optm
        self.num_e = num_e
        print(self.model)
    def _train(self):
        for j in range(self.num_e):
            for m, [x, y] in enumerate(self.train_loader):
                x, y = x.permute(1, 0, 2).float().to(self.device), y.float().to(self.device)
                out = self.model(x)
                out = f.sigmoid(out.squeeze())
                self.optm.zero_grad()
                loss = self.loss(out, f.sigmoid(y))
                loss.backward()
                self.optm.step()
                if (m % 100) == 0:
                    print('Epoch:{}, Iter: {}, Loss: {:.5f}'.format(j + 1, m + 1, loss))

    def train(self):
        return self._train()


trainer = trainer(train_loader, val_loader, model, loss, optimizer, num_epochs, device)
trainer.train()

