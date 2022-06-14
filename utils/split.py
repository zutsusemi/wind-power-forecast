import torch
import torch.nn.functional as F
import numpy as np
import os
import PIL.Image as Image
from torch.utils.data import DataLoader
import torchvision.transforms as t

class DatasetSampler:
    def __init__(self, length, num_test):
        self.length = length
        self.num_test = num_test
    def __call__(self, dataset):
        size = [self.length - self.num_test, self.num_test]
        return torch.utils.data.dataset.random_split(dataset, size)