from typing import Tuple
import pandas as pd
import numpy as np
from torch.utils import data
from typing import Tuple

def process_dataset(path: str, 
                    step: int = 30,
                    output_step: int = 144) -> Tuple[np.array, np.array]:
    """helper function to process the dataset

    Args:
        path (str): the path to the dataset
        step (int): the step of the dataset
    
    Returns:
        x (np.array): data of the dataset
        y (np.array): label of the dataset
    """
    train_val = pd.read_csv(path)
    turb_id = train_val['TurbID']
    wspd = train_val['Wspd']
    wdir = train_val['Wdir']
    etmp = train_val['Etmp']
    itmp = train_val['Itmp']
    Ndir = train_val['Ndir']
    Pab1 = train_val['Pab1']
    Pab2 = train_val['Pab2']
    Pab3 = train_val['Pab3']
    Prtv = train_val['Prtv']
    Patv = train_val['Patv']
    train_val_set = np.c_[turb_id,wspd,wdir,etmp,itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv]
    x, y = [], []
    for j in range(step, train_val_set.shape[0] - output_step):
        if train_val_set[j - step, 0] == train_val_set[j - 1, 0]:
            x.append(train_val_set[j - step : j, :])
            y.append(train_val_set[j : j + output_step, 10])
    return x, y


class Kddcup(data.Dataset):
    def __init__(self, 
                 path: str, 
                 step: int = 30,
                 out_step: int = 144):
        """constructor

        Args:
            path (str): the path to the dataset
            step (int): the step of the dataset
        """
        self.path = path
        self.step = step
        self._x, self._y = process_dataset(path, step, out_step)
    
    def __len__(self):
        return len(self._y)
    
    def __getitem__(self, index):
        return self._x[index], self._y[index]


def load(PATH = '../data/train_val.csv') -> data.Dataset:
    return Kddcup(PATH, step = 512, out_step = 144)