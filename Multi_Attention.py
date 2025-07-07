import math
import torch
import random
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_head
        

