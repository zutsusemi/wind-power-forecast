import torch
from torch.utils.data import DataLoader, Dataset

class Trainer:
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
                out = torch.sigmoid(out.squeeze())
                self.optm.zero_grad()
                loss = self.loss(out, torch.sigmoid(y))
                loss.backward()
                self.optm.step()
                if (m % 100) == 0:
                    print('Epoch:{}, Iter: {}, Loss: {:.5f}'.format(j + 1, m + 1, loss))

    def train(self):
        return self._train()

def train(device: torch.device,
          model: torch.nn.Module,
          train_set: Dataset,
          val_set: Dataset,
          batch_size: int, 
          lr: float, 
          epochs: int) -> None:
    """the training function

    Args:
        device (torch.device): device
        model (nn.Module): the model to be trained
        train_set (Dataset): the train set
        val_set (Dataset): the validation set
        batch_size (int): the batch size
        lr (float): the learning rate
        epochs (int): the number of training epochs

    Return:
        None
    """
    # construct the data loader
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, 1, shuffle=True)

    # construct the loss function and optimizer
    loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer = Trainer(train_loader, val_loader, model, loss, optimizer, epochs, device)
    trainer.train()