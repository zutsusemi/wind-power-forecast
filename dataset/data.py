import torch
import pandas as pd
import numpy as np
from torch.utils import data


def process_dataset(path, step = 30, output_step = 144):
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
    def __init__(self, path, step=30, out_step=144):
        self._x, self._y = process_dataset(path, step, out_step)
    
    def __len__(self):
        return len(self._y)
    
    def __getitem__(self, index):
        return self._x[index], self._y[index]


path  = './train_val.csv'


process_dataset(path)


