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
    speed = train_val['Wind Speed (m/s)'].to_numpy()
    power_curve = train_val['Theoretical_Power_Curve (KWh)'].to_numpy()
    active_power = train_val['LV ActivePower (kW)'].to_numpy()
    train_val_set = np.c_[speed, power_curve, active_power]
    x, y = [], []
    for j in range(step, train_val_set.shape[0] - output_step):
        x.append(train_val_set[j - step : j - 1, :])
        y.append(train_val_set[j : j + output_step, 2])
    return x, y


class Scada(data.Dataset):
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
    return Scada(PATH, step = 512, out_step = 144)