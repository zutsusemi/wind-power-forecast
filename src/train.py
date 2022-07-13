import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Trainer:
    def __init__(self, train_loader, val_loader, model, loss, optm, num_e, device, **kwargs):
        self.model = model
        self.train_loader  = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss, self.optm = loss, optm
        self.num_e = num_e
        self.load = None
        self.save = None
        self.s_iter = None
        if len(kwargs.keys()) > 0:
            self.load = kwargs['load']
            self.save = kwargs['save']
            self.s_iter = kwargs['siter']
        print(self.model)
        s=0
        for p in self.model.parameters():
            s += len(p.flatten())
        print('Total params: ', str(s))
    def _robustness(self):
        self.model.eval()
        var, count = 0, 0
        mean = 0
        with torch.no_grad():
            for m, [x, y] in tqdm(enumerate(self.val_loader)):
                x, y = x.float().to(self.device), y.float().to(self.device)
                o = self.model(x)
                o_2 = self.model(x+x*0.01)
                var += torch.sum((o - o_2)**2)
                mean += o.sum()
                count += len(o.flatten())
        var /= count
        mean /= count
        var = var ** 0.5
        return mean, var
    def _checkpoint(self, mode = 'save', **kwargs):
        if mode == 'save' and "path" in kwargs.keys():
            pth = kwargs['path']
            num_iter = kwargs['num_iter']
            os.makedirs(pth, exist_ok = True)
            torch.save(self.model.state_dict(), pth + 'model_weight'+'_'+str(num_iter) + ".pth")
            
            print('[logger] Checkpoint '+str(num_iter)+' saved.')
        elif mode == 'load' and "path" in kwargs.keys():
            pth = kwargs['path']
            self.model.load_state_dict(torch.load(pth))

            print('Loading checkpoint from ' + pth)
    
    def _evaluation(self, count=0, anime=False):
        self.model.eval()
        sum = 0
        N = 0
        import matplotlib.animation as animep
        import matplotlib.pyplot as plt
        metadata = dict(title='anime', artist='wasabi')
        writer = animep.PillowWriter(fps=15,metadata=metadata)
        fig_ = plt.figure()
        ax  = fig_.subplots()
        fp = 0
        with writer.saving(fig_, 'D:\\2021_Summer\\VE450\\models\\wind-power-forecast\\output\\ani.gif', 128):
            for m, [x, y] in tqdm(enumerate(self.val_loader)):
                x, y = x.float().to(self.device), y.float().to(self.device)
                # print(x.shape)
                with torch.no_grad():
                    out = self.model(x)
                # print(out.shape)
                out = out.squeeze(-1).permute(1,0,2)
                loss = ((out - y)**2).sum()
                
                sum += loss
                N += out.shape[0] * out.shape[1] * out.shape[2]

                x_ = x[0, :, 2, 10].detach().cpu().numpy() # (L, B, 1)
                y_ = y[0, :, 2].squeeze().detach().cpu().numpy()
                out_ = out[0, :, 2].squeeze().detach().cpu().numpy()


                
                if anime==True:
                    ax.plot(np.arange(x_.shape[0]+y_.shape[0]), np.concatenate([x_, y_]), label='gt')
                    ax.plot(np.arange(x_.shape[0]+y_.shape[0]), np.concatenate([x_, out_]), label='predict')
                    ax.legend()
                    ax.set_ylim(0,1000)
                    plt.xlabel('Timestamp')
                    plt.ylabel('Power')
                    ax.axvline(x=256, c='b',ls='--',lw=1.5)
                    writer.grab_frame()
                    # plt.pause(0.1)
                    plt.cla()
                    fp += 1
                    

                    if fp > 2048:
                        break
                    

                    
                    

                    
        
        sum /= N # (((y^ - y)**2)/N)**0.5
        sum = sum ** 0.5
        
        return sum.detach().cpu().numpy()
               
    def _train(self):
        count = 0
        if self.load is not None:
            self._checkpoint('load', path=self.load)
        for j in range(self.num_e):
            for m, [x, y] in enumerate(self.train_loader):
                x, y = x.float().to(self.device), y.float().to(self.device)
                out = self.model(x)
                out = out.squeeze(-1).permute(1,0,2)
                self.optm.zero_grad()
                loss = self.loss(out, y) / (out.shape[0] * out.shape[1] * out.shape[2])
                loss.backward()
                self.optm.step()
                if (m % 100) == 0:
                    print('Epoch:{}, Iter: {}, Tol Iter: {}, Loss: {:.5f}, RMSE: {:.5f}'.format(j + 1, m + 1, count + 1, loss, loss ** (1/2)))
                
                if self.save is not None:
                    if count % self.s_iter == 0:
                        print(self._evaluation(count, anime=True))
                        print(self._robustness())
                        self.model.train()
                        self._checkpoint('save', path = self.save, num_iter = count)

                count += 1
    

    def train(self):
        return self._train()

def train(device: torch.device,
          model: torch.nn.Module,
          train_set: Dataset,
          val_set: Dataset,
          batch_size: int, 
          lr: float, 
          epochs: int,
          load = None, 
          save = None,
          s_iter = None) -> None:
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
    val_loader = DataLoader(val_set, 1, shuffle=False)

    # construct the loss function and optimizer
    loss = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    trainer = Trainer(train_loader, val_loader, model, loss, optimizer, epochs, device, load = load, save = save, siter = s_iter)
    trainer.train()