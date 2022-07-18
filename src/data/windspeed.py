from typing import Tuple
import pandas as pd
import numpy as np
from torch.utils import data
from typing import Tuple

def process_dataset(path: str, 
                    step: int = 10,
                    output_step: int = 1) -> Tuple[np.array, np.array]:
    """helper function to process the dataset

    Args:
        path (str): the path to the dataset
        step (int): the step of the dataset
    
    Returns:
        x (np.array): data of the dataset
        y (np.array): label of the dataset
    """
    train_val = pd.read_excel(path)
    speed = train_val['sp'].to_numpy()
    x, y = [], []
    for j in range(step, train_val.shape[0] - output_step):
        x.append(speed[j - step: j])
        y.append(speed[j: j + output_step])
    return x, y

class WindSp(data.Dataset):
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
def load(PATH, step=5, out_step=1) -> data.Dataset:
    return WindSp(PATH, step, out_step)