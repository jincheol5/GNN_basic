import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EvolveGCN_H(nn.Module):
    def __init__(self,input_d):
        super().__init__()
        self.gru1=nn.GRU(input_size=input_d,hidden_size=input_d,num_layers=1,batch_first=True)
        self.gcn1=GCNConv()

    def forward(self):
        pass