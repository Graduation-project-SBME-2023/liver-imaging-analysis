#global imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adam(torch.optim.Adam):
    pass

class SGD(torch.optim.SGD):
    pass


class Optimizers():
    def __init__(self):
        self.optimizers={
            'Adam': Adam,
            'SGD':SGD
        }
    def choose(self, optimizer_name):
        return self.optimizers[optimizer_name]
